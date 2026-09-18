"""Optional MFCC and reference-informed ZTWCC, each 39 x 4 = 156 values."""
from __future__ import annotations

import numpy as np
from scipy import signal


def moments(values):
    """Population standardized moments, with defined zero-variance handling."""
    centered = values - values.mean(axis=-1, keepdims=True)
    variance = np.mean(centered**2, axis=-1)
    scale = np.max(np.abs(values), axis=-1)
    varying = variance > (np.finfo(float).eps * np.maximum(scale, np.finfo(float).tiny))**2
    skew = np.zeros(len(values))
    kurt = np.zeros(len(values))
    skew[varying] = np.mean(centered[varying]**3, axis=-1) / variance[varying]**1.5
    kurt[varying] = np.mean(centered[varying]**4, axis=-1) / variance[varying]**2
    return skew, kurt


def deltas(values):
    # [IMPLEMENTATION DECISION] regression +/-2 frames; edge replication.
    padded = np.pad(values, ((0, 0), (2, 2)), mode="edge")
    return sum(n * (padded[:, 2+n:2+n+values.shape[1]] -
                    padded[:, 2-n:2-n+values.shape[1]]) for n in (1, 2)) / 10


def aggregate(static):
    delta = deltas(static)
    values = np.vstack([static, delta, deltas(delta)])
    skew, kurt = moments(values)
    std = values.std(axis=1, ddof=1) if values.shape[1] > 1 else np.zeros(39)
    return np.concatenate([values.mean(axis=1), std, kurt, skew])


def mfcc(x):
    import librosa
    # [IMPLEMENTATION DECISION] 20 ms rectangular frames, hop=160,
    # 256-point FFT, 26 Slaney-normalized Mel filters, DCT-II, c0 included.
    frames = x[:len(x)//160*160].reshape(-1, 160)
    power = abs(np.fft.rfft(frames, n=256, axis=1))**2
    mel = librosa.filters.mel(sr=8000, n_fft=256, n_mels=26,
                             fmin=0, fmax=4000, htk=False, norm="slaney")
    logmel = librosa.power_to_db(mel @ power.T, ref=1.0, amin=1e-10, top_db=None)
    static = librosa.feature.mfcc(S=logmel, n_mfcc=13, dct_type=2, norm="ortho", lifter=0)
    return aggregate(static)


def ztwcc(x):
    # [PAPER] nFFT=128 and 13+13+13. [REFERENCE 35] 5 ms segments,
    # 6.25 ms hop, w1^2*w2, NGD, double differences, Hilbert envelope,
    # real IFFT(log spectrum). Other numerical choices are documented.
    x = signal.lfilter([1.0, -0.97], [1.0], x)
    n = np.arange(40)
    w1 = np.zeros(40)
    w1[1:] = 1 / (4 * np.sin(np.pi * n[1:] / (2 * 128))**2)
    window = w1**2 * 4 * np.cos(np.pi * n / (2 * 40))**2
    coefficients = []
    for start in range(0, len(x) - 40 + 1, 50):
        segment = x[start:start+40] * window
        spectrum = np.fft.fft(segment, 128)
        weighted = np.fft.fft(n * segment, 128)
        ngd = np.real(spectrum * np.conjugate(weighted))
        # [IMPLEMENTATION DECISION] centered periodic second difference.
        difference = np.roll(ngd, -1) - 2 * ngd + np.roll(ngd, 1)
        envelope = abs(signal.hilbert(difference))
        cepstrum = np.fft.ifft(np.log(np.maximum(envelope, 1e-12))).real
        coefficients.append(cepstrum[:13])
    return aggregate(np.asarray(coefficients).T)
