use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use num_complex::Complex32;
use rtl_sdr_rs::{RtlSdr, TunerGain};
use rustfft::FftPlanner;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, atomic::AtomicBool, mpsc},
};
use tracing::{error, info, warn};

const FFT_SIZE: usize = 1024;

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

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt::init();

    let mut exit = AtomicBool::new(false);

    // --- Set up RTL SDR device ---

    let mut device = RtlSdr::open_first_available().expect("Couldn't find/open device.");

    info!("Tuner ID: {:?}", device.get_tuner_id());
    info!("Tuner gains: {:?}", device.get_tuner_gains());
    info!("Sample rate: {:?}", device.get_sample_rate());
    info!("Center freq: {:?}", device.get_center_freq());
    info!("Frequency correction: {:?}", device.get_freq_correction());

    // Set sane defaults
    device.set_center_freq(90_700_000)?;
    device.set_sample_rate(2_400_000)?;
    device.set_tuner_gain(TunerGain::Auto)?;

    let mut device_buffer = vec![0u8; 16384];

    device.reset_buffer()?;

    // --------------------
    // --- Set up AUDIO ---
    // --------------------

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

    // Spawn READER thread
    let (iq_tx, iq_rx) = mpsc::sync_channel::<Vec<u8>>(32);

    let audio_buf_write_clone = audio_buf_write.clone();

    // Receives raw IQ samples from the main thread loop below
    let audio_processor = std::thread::spawn(move || {
        // Low pass FILTERS
        let mut lpf1 = LowPassFilter::new(64, 0.05);
        let mut lpf2 = LowPassFilter::new(32, 0.083);

        // Previous sample for demodulation
        let mut prev = Complex32::new(0.0, 0.0);

        while let Ok(raw) = iq_rx.recv() {
            // -----------
            // --- FFT ---
            // -----------

            // Convert IQ pairs to one complex number I + Qi
            // Also normalise and center
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
            let mut buf = audio_buf_write_clone.lock().unwrap();

            if buf.len() > 48000 {
                warn!("Audio buffer backing up: {} samples", buf.len());
            }

            if buf.len() < 48000 * 2 {
                buf.extend(audio.iter());
            }
        }
    });

    // ------------------
    // --- Set up FFT ---
    // ------------------

    // let mut planner = FftPlanner::new();
    // let fft = planner.plan_fft_forward(FFT_SIZE);

    let hann_window: Vec<f32> = (0..FFT_SIZE)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / FFT_SIZE as f32;
            x.sin().powi(2)
        })
        .collect();

    // -------------
    // --- READ! ---
    // -------------

    // Read IQ samples from RTL SDR
    while !exit.load(std::sync::atomic::Ordering::Relaxed) {
        device.read_sync(&mut device_buffer)?;

        // Send raw IQ sampled to audio thread
        if iq_tx.send(device_buffer.clone()).is_err() {
            break;
        }
    }

    device.close()?;

    Ok(())
}
