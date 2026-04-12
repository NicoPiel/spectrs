# spectrs 📻

A real-time, terminal-based Software Defined Radio (SDR) and FM demodulator written in Rust.

`spectrs` leverages an RTL-SDR to capture raw RF signals, processes them through a custom Digital Signal Processing (
DSP) pipeline, and renders a live spectrum analyzer and waterfall display directly in your terminal using `ratatui`.

## Features

* **Terminal UI:** Live FFT spectrogram and scrolling color-mapped waterfall display.
* **Real-Time DSP:** Custom multi-stage decimation, FM demodulation, and European (50µs) de-emphasis filtering.
* **Live Audio:** Streams demodulated 48kHz audio directly to your system's default output.
* **On-the-fly Tuning:** Adjust the center frequency without dropping the audio stream or resetting the hardware.

## Prerequisites

You will need an RTL-SDR USB dongle and the appropriate C libraries installed on your system to compile the hardware
bindings.

**Linux (Debian/Ubuntu):**

```bash
sudo apt-get install librtlsdr-dev libasound2-dev
```

*(Note: `libasound2-dev` is required for ALSA support via the `cpal` audio backend).*

**macOS:**

```bash
brew install librtlsdr
```

## Installation

Clone the repository and build using Cargo:

```bash
git clone [https://github.com/yourusername/spectrs.git](https://github.com/yourusername/spectrs.git)
cd spectrs
cargo build --release
```

## Usage

Plug in your RTL-SDR dongle and run the application. (Running in `--release` mode is highly recommended to ensure the
DSP thread can keep up with the 2.4 MSPS data rate).

```bash
cargo run --release
```

### Controls

* <kbd>Up Arrow</kbd> : Tune frequency up by 100 kHz.
* <kbd>Down Arrow</kbd> : Tune frequency down by 100 kHz.
* <kbd>q</kbd> : Quit application gracefully.

## The DSP Pipeline

For those interested in the radio math, the audio pipeline processes data in the following stages:

1. **Hardware Capture:** Reads raw Complex IQ samples from the RTL-SDR at 2.4 MSPS.
2. **RF Channelization:** Applies a windowed-sinc low-pass filter and decimates the signal by a factor of 10, isolating
   a clean 240 kHz slice of the spectrum.
3. **FM Demodulation:** Calculates the phase difference between consecutive complex samples to extract the
   frequency-modulated audio.
4. **De-emphasis:** Applies a 50µs recursive filter (European standard) to attenuate high-frequency transmission noise
   and hiss.
5. **Audio Decimation:** Applies a final low-pass filter and decimates the audio by a factor of 5 down to the standard
   48 kHz sample rate.
6. **Playback:** Pushed to a lock-free ring buffer consumed by the `cpal` audio stream.

## Logging

To prevent terminal UI corruption, logging (`stdout`) is routed to a black-hole sink by default. If you are developing
and need to see `tracing` output, you can redirect the `tracing_appender` to a file in `main.rs`.

## License

MIT