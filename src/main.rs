use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{Event, KeyCode};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{event, ExecutableCommand};
use num_complex::Complex32;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::widgets::{Block, Borders};
use ratatui::Terminal;
use rtl_sdr_rs::{RtlSdr, TunerGain};
use rustfft::FftPlanner;
use std::io::stdout;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::{
    collections::VecDeque,
    sync::{atomic::AtomicBool, mpsc, Arc, Mutex},
};
use tokio::sync::broadcast;
use tracing::{error, info, warn};

const FFT_SIZE: usize = 1024;
const DEFAULT_FREQUENCY: u32 = 90_700_000;

struct AppState {
    current_frequency: u32,
    current_gain: TunerGain,
    latest_fft: Vec<f32>,
    waterfall_history: VecDeque<Vec<f32>>,
}

struct LowPassFilter {
    coeffs: Vec<f32>,
    history: VecDeque<f32>,
}

impl LowPassFilter {
    fn new(num_taps: usize, cutoff: f32) -> Self {
        let coeffs: Vec<f32> = (0..num_taps)
            .map(|i| {
                let n = i as f32 - (num_taps - 1) as f32 / 2.0;
                let sinc = if n.abs() < 1e-6 {
                    2.0 * cutoff
                } else {
                    (2.0 * std::f32::consts::PI * cutoff * n).sin() / (std::f32::consts::PI * n)
                };

                let window = (std::f32::consts::PI * i as f32 / (num_taps - 1) as f32)
                    .sin()
                    .powi(2);

                sinc * window
            })
            .collect();

        LowPassFilter {
            coeffs,
            history: VecDeque::from(vec![0.0; num_taps]),
        }
    }

    fn process_and_decimate(&mut self, input: &[f32], decimation: usize) -> Vec<f32> {
        let mut output = Vec::with_capacity(input.len() / decimation + 1);

        for (i, &sample) in input.iter().enumerate() {
            self.history.pop_front();
            self.history.push_back(sample);

            if i % decimation == 0 {
                let sum: f32 = self
                    .history
                    .iter()
                    .zip(&self.coeffs)
                    .map(|(s, c)| s * c)
                    .sum();
                output.push(sum);
            }
        }

        output
    }
}

enum SdrCommand {
    TuneUp,
    TuneDown,
    GainUp,
    GainDown,
}

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let (non_blocking, _guard) = tracing_appender::non_blocking(std::io::sink());
    tracing_subscriber::fmt().with_writer(non_blocking).init();

    let mut state = AppState {
        current_frequency: DEFAULT_FREQUENCY,
        current_gain: TunerGain::Auto,
        latest_fft: vec![0.0; FFT_SIZE],
        waterfall_history: VecDeque::new(),
    };

    static SHUTDOWN: AtomicBool = AtomicBool::new(false);

    ctrlc::set_handler(|| {
        SHUTDOWN.swap(true, Ordering::Relaxed);
    })?;

    let (fft_watch_tx, fft_watch_rx) = tokio::sync::watch::channel::<Vec<f32>>(vec![0.0; 32]);

    // Spawn READER thread
    let (iq_tx, _) = broadcast::channel::<Arc<Vec<u8>>>(32);
    let audio_rx = iq_tx.subscribe();
    let fft_rx = iq_tx.subscribe();

    let audio_processor = tokio::spawn(async move {
        if let Err(e) = audio(audio_rx, &SHUTDOWN).await {
            error!("Audio tasked failed: {e}");
        }
    });

    let fft_processor = tokio::spawn(async move {
        if let Err(e) = fft(fft_rx, fft_watch_tx, &SHUTDOWN).await {
            error!("FFT task failed: {e}");
        }
    });

    // -------------
    // --- READ! ---
    // -------------

    let (command_tx, command_rx) = mpsc::channel::<SdrCommand>();

    let receive_processor =
        tokio::task::spawn_blocking(|| receive_iq(iq_tx, command_rx, &SHUTDOWN));

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;

    // Read IQ samples from RTL SDR

    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let tui_result = tui(
        &mut terminal,
        command_tx,
        fft_watch_rx,
        &mut state,
        &SHUTDOWN,
    );

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;

    tui_result?;

    let _ = receive_processor.await;

    Ok(())
}

fn receive_iq(
    iq_tx: broadcast::Sender<Arc<Vec<u8>>>,
    command_rx: mpsc::Receiver<SdrCommand>,
    shutdown: &AtomicBool,
) -> color_eyre::Result<()> {
    // --- Set up RTL SDR device ---
    let mut device = RtlSdr::open_first_available().expect("Couldn't find/open device.");

    info!("Tuner ID: {:?}", device.get_tuner_id());
    info!("Tuner gains: {:?}", device.get_tuner_gains());
    info!("Sample rate: {:?}", device.get_sample_rate());
    info!("Center freq: {:?}", device.get_center_freq());
    info!("Frequency correction: {:?}", device.get_freq_correction());

    // Set sane defaults
    let mut current_frequency = DEFAULT_FREQUENCY;
    device.set_center_freq(current_frequency)?;
    device.set_sample_rate(2_400_000)?;
    device.set_tuner_gain(TunerGain::Auto)?;

    // Necessary for RTL SDR to work
    device.reset_buffer()?;

    let mut device_buffer = vec![0u8; 16384];

    info!("Trying recv..");

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        while let Ok(cmd) = command_rx.try_recv() {
            match cmd {
                SdrCommand::TuneUp => {
                    current_frequency += 100_000;
                    device.set_center_freq(current_frequency)?;
                }
                SdrCommand::TuneDown => {
                    current_frequency -= 100_000;
                    device.set_center_freq(current_frequency)?;
                }
                SdrCommand::GainUp => {}
                SdrCommand::GainDown => {}
            }
        }

        device
            .read_sync(&mut device_buffer)
            .expect("Error reading from device buffer");

        // Send raw IQ sampled to audio thread
        iq_tx
            .send(Arc::new(device_buffer.clone()))
            .expect("Error sending IQ buffer.");
    }

    Ok(())
}

fn tui(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    command_tx: mpsc::Sender<SdrCommand>,
    mut fft_rx: tokio::sync::watch::Receiver<Vec<f32>>,
    state: &mut AppState,
    shutdown: &AtomicBool,
) -> color_eyre::Result<()> {
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        if fft_rx.has_changed()? {
            let new_data = fft_rx.borrow_and_update().clone();
            state.latest_fft = new_data;
        }

        terminal.draw(|frame| {
            let area = frame.area();

            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(area);

            let spectrogram_block = Block::default()
                .title(format!(" Spectrogram | {} Hz ", state.current_frequency))
                .borders(Borders::ALL);
            frame.render_widget(spectrogram_block, chunks[0]);

            let waterfall_block = Block::default().title(" Waterfall ").borders(Borders::ALL);
            frame.render_widget(waterfall_block, chunks[1]);
        })?;

        if event::poll(Duration::from_millis(16))?
            && let Event::Key(key) = event::read()?
        {
            match key.code {
                KeyCode::Char('q') => {
                    shutdown.store(true, Ordering::Relaxed);
                    break;
                }
                KeyCode::Up => {
                    state.current_frequency += 100_000;
                    let _ = command_tx.send(SdrCommand::TuneUp);
                }
                KeyCode::Down => {
                    state.current_frequency -= 100_000;
                    let _ = command_tx.send(SdrCommand::TuneDown);
                }
                _ => {}
            }
        }
    }

    Ok(())
}

// --------------------
// --- Set up AUDIO ---
// --------------------
async fn audio(
    mut audio_rx: broadcast::Receiver<Arc<Vec<u8>>>,
    shutdown: &AtomicBool,
) -> color_eyre::Result<()> {
    info!("Setting up audio..");
    // Set up ring buffer
    let audio_buf = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let audio_buf_write = audio_buf.clone();

    // Setup audio devices
    let audio_host = cpal::default_host();
    let audio_device = audio_host
        .default_output_device()
        .expect("Couldn't find output device");
    let audio_config = cpal::StreamConfig {
        channels: 1,
        sample_rate: 48_000,
        buffer_size: cpal::BufferSize::Default,
    };

    // Build audio output stream
    let stream = audio_device
        .build_output_stream(
            &audio_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut buf = audio_buf.lock().unwrap();

                for sample in data.iter_mut() {
                    *sample = buf.pop_front().unwrap_or(0.0);
                }
            },
            |err| error!("Audio error: {err}"),
            None,
        )
        .expect("Failed to build audio stream.");

    // Now play audio
    stream.play().expect("Failed to play audio stream.");

    // Low pass filters
    let mut lpf1 = LowPassFilter::new(64, 0.05);
    let mut lpf2 = LowPassFilter::new(32, 0.083);

    // Previous sample for demodulation
    let mut prev = Complex32::new(0.0, 0.0);

    while let Ok(raw) = audio_rx.recv().await {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        // Convert IQ pairs to one complex number I + Qi
        // Also normalize and center
        let iq: Vec<Complex32> = raw
            .chunks_exact(2)
            .map(|pair| {
                Complex32::new(
                    (pair[0] as f32 - 127.5) / 127.5,
                    (pair[1] as f32 - 127.5) / 127.5,
                )
            })
            .collect();

        // -------------
        // --- AUDIO ---
        // -------------

        // Demodulate IQ for audio output
        // Sampling at default rate gives us 2.4 MHz
        let demodulated: Vec<f32> = iq
            .iter()
            .map(|&sample| {
                let product = sample * prev.conj();
                prev = sample;
                product.arg()
            })
            .collect();

        // Low pass stage 1
        let stage1 = lpf1.process_and_decimate(&demodulated, 10); // default 2.4MHz -> 240kHz
        let stage2 = lpf2.process_and_decimate(&stage1, 5); // default 240kHz -> 48kHz

        let audio: Vec<f32> = stage2.iter().map(|&s| s * 0.3).collect();

        /*info!(
            "iq: {}, demod: {}, stage1: {}, audio: {}, buf: {}",
            iq.len(),
            demodulated.len(),
            stage1.len(),
            audio.len(),
            audio_buf_write_clone.lock().unwrap().len()
        );*/

        // Write to audio ring buffer
        let mut buf = audio_buf_write.lock().unwrap();

        if buf.len() > 48000 {
            warn!("Audio buffer backing up: {} samples", buf.len());
        }

        if buf.len() < 48000 * 2 {
            buf.extend(audio.iter());
        }
    }

    Ok(())
}

// ------------------
// --- Set up FFT ---
// ------------------
async fn fft(
    mut fft_rx: broadcast::Receiver<Arc<Vec<u8>>>,
    fft_watch_tx: tokio::sync::watch::Sender<Vec<f32>>,
    shutdown: &AtomicBool,
) -> color_eyre::Result<()> {
    info!("Setting up FFT..");
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let hann_window: Vec<f32> = (0..FFT_SIZE)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / FFT_SIZE as f32;
            x.sin().powi(2)
        })
        .collect();

    // --- PRE-ALLOCATE BUFFERS HERE ---
    let mut frame = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
    let mut magnitudes = vec![0.0; FFT_SIZE];

    while let Ok(raw) = fft_rx.recv().await {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let iq: Vec<Complex32> = raw
            .chunks_exact(2)
            .map(|pair| {
                Complex32::new(
                    (pair[0] as f32 - 127.5) / 127.5,
                    (pair[1] as f32 - 127.5) / 127.5,
                )
            })
            .collect();

        for chunk in iq.chunks_exact(FFT_SIZE) {
            // Apply window and copy into pre-allocated frame
            for (i, (&sample, &window)) in chunk.iter().zip(&hann_window).enumerate() {
                frame[i] = sample * window;
            }

            fft.process(&mut frame);

            // Calculate magnitudes and apply FFT Shift simultaneously
            let half_size = FFT_SIZE / 2;
            for i in 0..FFT_SIZE {
                // Shift: indices 0..511 go to 512..1023, and 512..1023 go to 0..511
                let shifted_i = if i < half_size {
                    i + half_size
                } else {
                    i - half_size
                };

                // Add a tiny epsilon (1e-10) to avoid log10(0) if a bin is perfectly empty
                let power = frame[i].norm_sqr() + 1e-10;
                magnitudes[shifted_i] = 10.0 * power.log10();
            }

            let _ = fft_watch_tx.send(magnitudes.clone())?;
        }
    }

    Ok(())
}
