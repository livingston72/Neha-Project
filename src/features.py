"""One feature extractor shared by dataset processing, validation and demo."""
from __future__ import annotations

import csv
import json
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .banks import BANKS, FILTER_COUNTS, FS, cqt_design
from .cepstra import mfcc, moments, ztwcc
from .common import KEYS, feature_path, log_error, save_json, sha256, stem
from .dataset import AudioError, read_audio

DIMENSIONS = {"CQT-LTAS": 1069, "GT-LTAS": 329, "SFF-LTAS": 2019,
              "LTAS": 99, "MFCC": 156, "ZTWCC": 156}
FEATURE_VERSION = "ltas-reference-informed-v1"


def specification(feature="CQT-LTAS"):
    return {"version": FEATURE_VERSION, "feature_type": feature, "dimension": DIMENSIONS[feature],
            "sample_rate": FS, "mono": "arithmetic channel mean", "resampling": "scipy.resample_poly",
            "resample_window": ["kaiser", 5.0], "resample_padtype": "constant",
            "frame_samples": 160, "frame_hop": 160, "frame_window": "rectangular",
            "partial_frame": "drop trailing incomplete frame; reject clips shorter than 20ms",
            "waveform_normalization": "none", "ltas_order": "reference statistic-major",
            "std_ddof": 1, "skew_bias": True, "kurtosis": "Pearson, bias=True",
            "constant_moments": 0.0, "zero_denominator_result": 0.0,
            "cqt": {"fmin": 10, "fmax": 4000, "bins_per_octave": 12,
                    "interior_centers": 104, "boundary_filters": [0, 4000], "total_filters": 106,
                    "construction": "LTFAT-informed continuous frequency prototypes; periodic convolution",
                    "filter_prototype": "frequency-domain Hann; DC/Nyquist flat-top taper",
                    "minimum_support_fft_bins": 4, "gain": "sqrt(N/ceil(N*support/fs)); endpoints /sqrt(2)"},
            "gt": {"filters": 32, "range_hz": [0, 4000], "centers": "ERB-rate spaced",
                   "order": 4, "phase": 0, "bandwidth": "24.7*(4.37*fc/1000+1)",
                   "impulse_decay_limit": 30, "gain": "unit magnitude at center"},
            "sff": {"filters": 201, "range_hz": [0, 4000], "step_hz": 20, "a": 0.98},
            "ltas_baseline": {"status": "REFERENCE PROXY, identity NOT SPECIFIED IN PAPER",
                              "fir_order": 32, "edges_hz": [0,30,60,120,240,480,960,1920,2400,3840]},
            "mfcc": {"coefficients": 13, "delta": 13, "double_delta": 13,
                     "nfft": 256, "mels": 26, "dct": 2, "include_c0": True,
                     "mel_norm": "slaney", "db_floor": 1e-10},
            "ztwcc": {"nfft": 128, "segment_samples": 40, "hop_samples": 50,
                      "preemphasis": 0.97, "coefficients": 13, "include_c0": True,
                      "log_floor": 1e-12, "second_difference": "centered periodic"},
            "delta_width": 5, "delta_edges": "replicate",
            "methodology": "See METHODOLOGY.md: PAPER / REFERENCE / IMPLEMENTATION DECISION"}


def preprocess(path: Path):
    x, sr, meta = read_audio(path)
    if sr != FS:
        divisor = gcd(sr, FS)
        x = resample_poly(x, FS // divisor, sr // divisor, window=("kaiser", 5.0), padtype="constant")
    if len(x) < 160:
        raise AudioError("short_audio", "Fewer than 160 samples after 8 kHz resampling")
    if not np.isfinite(x).all():
        raise AudioError("invalid_audio", "Non-finite samples after resampling")
    meta.update({"resampled_sample_rate": FS, "resampled_samples": len(x),
                 "resampled_duration": len(x)/FS, "frame_size": 160,
                 "number_of_frames": len(x)//160, "discarded_tail_samples": len(x)%160})
    return x, meta


def safe_divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=np.abs(denominator) > np.finfo(float).tiny)


def ltas_statistics(rms_matrix, band_rms):
    """[REFERENCE] LTAS_feat_CQT.m feat1..feat10, full-band component first.

    Remove ONLY full-band/full-band ratio; never drop an arbitrary last value.
    band_rms is calculated over the complete signal, including its trailing tail.
    """
    full = band_rms[0]
    mean = rms_matrix.mean(axis=1)
    std = rms_matrix.std(axis=1, ddof=1) if rms_matrix.shape[1] > 1 else np.zeros(len(band_rms))
    skew, kurt = moments(rms_matrix)
    span = np.ptp(rms_matrix, axis=1)
    pvi = np.mean(abs(np.diff(rms_matrix, axis=1)), axis=1) if rms_matrix.shape[1] > 1 else np.zeros(len(band_rms))
    return np.concatenate([safe_divide(band_rms[1:], full), safe_divide(mean, full), std,
                           safe_divide(std, full), safe_divide(std, band_rms), skew, kurt,
                           span, safe_divide(span, full), safe_divide(pvi, full)])


def extract(path: Path, feature="CQT-LTAS"):
    x, meta = preprocess(path)
    if feature in BANKS:
        rms = []
        band_rms = []
        def consume(component):
            if len(component) != len(x):
                raise ValueError("Filter changed signal length")
            power = abs(component)**2  # complex filter outputs need squared magnitude
            rms.append(np.sqrt(power[:len(x)//160*160].reshape(-1,160).mean(axis=1)))
            band_rms.append(np.sqrt(power.mean()))
        consume(x)
        for component in BANKS[feature](x):
            consume(component)
        rms = np.asarray(rms)
        expected = FILTER_COUNTS[feature] + 1
        if rms.shape != (expected, len(x)//160):
            raise ValueError(f"Unexpected RMS shape {rms.shape}")
        vector = ltas_statistics(rms, np.asarray(band_rms))
        meta.update({"number_of_filters": expected-1, "components": expected,
                     "filtered_signal_shape": [expected-1, len(x)],
                     "rms_matrix_shape": list(rms.shape), "statistics_per_component": 10})
    elif feature == "MFCC":
        vector = mfcc(x)
    elif feature == "ZTWCC":
        vector = ztwcc(x)
    else:
        raise ValueError(f"Unsupported feature type: {feature}")
    meta.update({"feature_type": feature, "final_feature_shape": list(vector.shape),
                 "nan_count": int(np.isnan(vector).sum()), "infinity_count": int(np.isinf(vector).sum())})
    if vector.shape != (DIMENSIONS[feature],) or not np.isfinite(vector).all():
        raise AudioError("invalid_features", f"Shape/finite check failed: {meta}")
    return vector, meta


def validate_clip(path: Path, feature="CQT-LTAS"):
    vector, report = extract(path, feature)
    print(json.dumps(report, indent=2))
    return vector, report


def extract_dataset(root: Path, target="block", feature="CQT-LTAS", dataset="sep28k"):
    selected = root / "data" / f"{stem(target, dataset)}_valid.csv"
    if not selected.exists():
        raise FileNotFoundError(f"Run prepare first: {selected}")
    df = pd.read_csv(selected, dtype={k: str for k in KEYS})
    if df.empty:
        raise ValueError("No valid selected clips; inspect processing_errors.csv and the WAV template")
    # Mandatory gate BEFORE opening the full feature output.
    try:
        _, validation = validate_clip(Path(df.iloc[0].audio_path), feature)
    except (AudioError, ValueError, RuntimeError, OSError) as exc:
        log_error(root, df.iloc[0].audio_path, getattr(exc, "kind", "single_clip_validation_error"), str(exc))
        raise
    output = feature_path(root, target, feature, dataset)
    save_json(output.with_suffix(".validation.json"), validation)
    temporary = output.with_suffix(".partial.csv")
    processed, failed = 0, 0
    distribution = {"0": 0, "1": 0}
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(KEYS + ["label"] + [f"feature_{i}" for i in range(1,DIMENSIONS[feature]+1)])
        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                vector, _ = extract(Path(row.audio_path), feature)
                writer.writerow([row[k] for k in KEYS] + [int(row.label)] + vector.tolist())
                distribution[str(int(row.label))] += 1
                processed += 1
            except (AudioError, ValueError, RuntimeError, OSError, FloatingPointError) as exc:
                failed += 1
                log_error(root, row.audio_path, getattr(exc, "kind", "feature_extraction_error"), str(exc))
            if i % 100 == 0 or i == len(df):
                print(f"Feature extraction: {i} / {len(df)} ({failed} failed)", flush=True)
    if processed == 0:
        raise ValueError("All clips failed; partial file retained, no successful feature file created")
    temporary.replace(output)
    report = {"total_clips": len(df), "processed_clips": processed, "failed_clips": failed,
              "feature_matrix_shape": [processed, DIMENSIONS[feature]],
              "nan_count": 0, "infinity_count": 0, "class_distribution": distribution,
              "dataset": dataset, "stutter_type": target, "feature_config": specification(feature),
              "selection_sha256": sha256(selected), "feature_csv_sha256": sha256(output)}
    save_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))
    return output
