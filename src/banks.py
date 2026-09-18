"""Filter banks. Provenance and numerical conventions: METHODOLOGY.md."""
from __future__ import annotations

import numpy as np
from scipy import signal

FS = 8000


def cqt_design(length: int):
    """[REFERENCE] LTFAT 2.4.0: 104 log centers plus DC/Nyquist fillers.

    Frequency-domain Hann prototypes are filter responses, never framing windows.
    Continuous evaluation on the DFT grid is an explicitly documented Python
    implementation decision; this is not a bit-for-bit MATLAB/LTFAT port.
    """
    inner = 10.0 * 2.0 ** (np.arange(104) / 12.0)
    centers = np.r_[0.0, inner, FS / 2]
    supports = np.r_[20.0, inner * (2 ** (1 / 12) - 2 ** (-1 / 12)),
                     2 * (FS / 2 - inner[-1])]
    supports = np.maximum(supports, 4 * FS / length)
    gains = np.sqrt(length / np.ceil(length * supports / FS))
    gains[[0, -1]] /= np.sqrt(2)
    return centers, supports, gains


def cqt_responses(length: int):
    centers, supports, gains = cqt_design(length)
    grid = np.fft.fftfreq(length, 1 / FS)
    for index, (center, width, gain) in enumerate(zip(centers, supports, gains)):
        distance = np.abs((grid - center + FS / 2) % FS - FS / 2)
        taper = 1.0
        if index == 0:
            taper = min(1.0, supports[1] / width)
        elif index == len(centers) - 1:
            taper = min(1.0, supports[-2] / width)
        flat = width * (1 - taper) / 2
        phase = np.clip((distance - flat) / (width * taper / 2), 0, 1)
        response = gain * (0.5 + 0.5 * np.cos(np.pi * phase))
        response[distance >= width / 2] = 0
        yield response


def iter_cqt(x):
    """N-point circular convolution with h=IFFT(H), as a periodic DFT bank."""
    spectrum = np.fft.fft(x)
    for response in cqt_responses(len(x)):
        yield np.fft.ifft(spectrum * response)


def iter_gammatone(x):
    # [IMPLEMENTATION DECISION] 32 ERB-rate-spaced centers including endpoints.
    # [PAPER] Eq. (2), order 3--5; choose 4. Eq. (3) uses fc in kHz.
    erb = np.linspace(0, np.log1p(0.00437 * 4000), 32)
    for center in np.expm1(erb) / 0.00437:
        bandwidth = 24.7 * (4.37 * center / 1000 + 1)
        length = int(np.ceil(30 * FS / (2 * np.pi * bandwidth))) + 1
        t = np.arange(length) / FS
        h = t**3 * np.exp(-2 * np.pi * bandwidth * t) * np.cos(2 * np.pi * center * t)
        # [IMPLEMENTATION DECISION] normalize transfer magnitude at center to 1.
        gain = abs(np.sum(h * np.exp(-2j * np.pi * center * t)))
        if gain <= np.finfo(float).tiny:
            raise ValueError("Degenerate gammatone kernel")
        yield signal.fftconvolve(x, h / gain, mode="full")[:len(x)]


def iter_sff(x):
    # [PAPER] H(z)=1/(1-a_k z^-1), a_k=.98 exp(-j omega_k).
    for center in np.arange(0, 4001, 20):
        pole = 0.98 * np.exp(-2j * np.pi * center / FS)
        yield signal.lfilter([1.0], [1.0, -pole], x)


def iter_ltas_baseline(x):
    # [REFERENCE] LTAS_feat.m, 8 kHz branch; baseline identity in target paper
    # is NOT SPECIFIED IN PAPER. This is a reference-based 99-D proxy.
    edges = [0, 30, 60, 120, 240, 480, 960, 1920, 2400, 3840]
    for i in range(9):
        low = 0.1 if i == 0 else edges[i] * 0.95
        high = edges[i + 1]
        h = signal.firwin2(33, [0, low, high, high * 1.01, 4000],
                          [0, 0, 1, 0, 0], fs=FS)
        # Filter design's Hamming taper is not the LTAS framing window.
        yield signal.fftconvolve(x, h, mode="full")[16:16 + len(x)]


BANKS = {"CQT-LTAS": iter_cqt, "GT-LTAS": iter_gammatone,
         "SFF-LTAS": iter_sff, "LTAS": iter_ltas_baseline}
FILTER_COUNTS = {"CQT-LTAS": 106, "GT-LTAS": 32, "SFF-LTAS": 201, "LTAS": 9}
