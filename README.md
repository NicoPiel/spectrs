# spectrs

## Plan

Antenne
  ↓ (Elektromagnetische Wellen)
RTL-SDR
  ↓ (uint8 Paare: I,Q,I,Q,I,Q...)
Normalisierung
  ↓ (Complex<f32> Stream)
  ├──→ FFT-Branch (für Display)
  │     ↓ Fensterfunktion (Hann)
  │     ↓ FFT (1024 Samples → 1024 Bins)
  │     ↓ Magnitude: sqrt(re² + im²)
  │     ↓ Dezibel: 20 * log10(mag)
  │     ↓ FFT-Shift (Center-Freq in die Mitte)
  │     ↓ → Ratatui Spektrum-Chart
  │     ↓ → Wasserfall (Zeilen nach unten schieben)
  │
  └──→ Audio-Branch (für Lautsprecher)
        ↓ FM: atan2(s[n] * conj(s[n-1]))
        ↓ AM: sqrt(I² + Q²)
        ↓ Tiefpassfilter (< 24 kHz)
        ↓ Dezimation (jeden N-ten behalten)
        ↓ → cpal Audio-Output

## License

Copyright (c) Nico Piel <nico.piel@hotmail.de>

This project is licensed under the MIT license ([LICENSE] or <http://opensource.org/licenses/MIT>)

[LICENSE]: ./LICENSE
