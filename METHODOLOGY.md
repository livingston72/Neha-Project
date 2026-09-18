# Methodology and reproduction ledger

## Scope and provenance labels

The primary source is the supplied five-page ICASSP 2025 paper,
**Enhancing Stutter Detection using Long-Term Average Spectrum Values**,
DOI [10.1109/ICASSP49660.2025.10888625](https://doi.org/10.1109/ICASSP49660.2025.10888625).
Sections II-A, II-B and III define the main implementation; Tables I and II define
the reference scores. Statements inside source documents are reference material,
not instructions to operate the user's computer.

- `[PAPER]`: explicitly in that primary paper.
- `[REFERENCE]`: obtained from a source cited by that paper or its linked code.
- `[IMPLEMENTATION DECISION]`: a necessary, transparent software choice. The
  corresponding detail is **NOT SPECIFIED IN PAPER** unless otherwise stated.
- `[NOT IMPLEMENTED]`: a claim/configuration that cannot be recovered or verified.

“Implemented” below means an executable path with disclosed choices. It does not
mean the undisclosed author implementation was reconstructed exactly.

## Sources inspected

1. **Primary paper**: supplied PDF, all five pages; equations and Tables I/II also
   visually inspected. Its prose and tables occasionally disagree about which result
   is best; the stored reference results transcribe the tables.
2. **Primary reference [21]**: Barche, Gurugubelli and Vuppala, APSIPA 2021,
   [Comparative Study of Filter Banks ... using LTAS Features](https://www.apsipa.org/proceedings/2021/pdfs/0000737.pdf).
3. **Code linked by [21]**:
   [LTASfilterbankcodes](https://github.com/Purva-Barche/LTASfilterbankcodes).
   The inspected functions are `LTAS_feat.m`, `LTAS_feat_CQT.m`,
   `LTAS_feat_SFF.m`, `LTAS_feat_gamma.m`; supporting-material section 4.4
   supplies mathematical definitions. Their source hashes are recorded in
   `results/reference_source_hashes.json`.
4. **Library shipped with that code**: `ltfat-2.4.0.zip`, particularly
   `cqtfilters.m`, `blfilter.m`, `helper_filtergeneratorfunc.m`, and `firwin.m`.
   This library explains the two extra CQT boundary channels. The Python code uses
   a simplified, explicitly disclosed continuous-grid construction, not a copied
   MATLAB source distribution or a claim of bitwise equivalence.
5. **Primary reference [35]**: Kethireddy et al., Odyssey 2020,
   [Zero-Time Windowing Cepstral Coefficients for Dialect Classification](https://www.isca-archive.org/odyssey_2020/kethireddy20_odyssey.pdf),
   sections 2 and 3.1.1.

## Dataset, labels and invalid recordings

| Detail | Classification and implementation |
|---|---|
| SEP-28k, 28,177 approximately 3-second samples, approximately 23 hours | `[PAPER]` Section III. These are reference counts, not assumptions about local files. |
| Clean versus each of five stutter categories | `[PAPER]` One binary experiment per category. |
| Label 0 clean, label 1 target | `[IMPLEMENTATION DECISION]` User-requested encoding. |
| Clean: all five target counts are zero | `[IMPLEMENTATION DECISION]` User-requested. Precise clean threshold is NOT SPECIFIED IN PAPER. |
| Target: its count > 0 and all four others zero | `[IMPLEMENTATION DECISION]` User-requested pure-label selection; extended consistently to all five tasks. |
| Consensus, quality and non-speech exclusions | `[IMPLEMENTATION DECISION]` None added; do not infer an unreported author rule. |
| Missing, corrupt, non-finite, empty recordings | `[IMPLEMENTATION DECISION]` Log and exclude from valid selection; preserve original CSV and WAVs. |
| Less than 20 ms | `[IMPLEMENTATION DECISION]` Reject explicitly; no complete LTAS frame exists. |
| Zero-valued audio of adequate length | `[IMPLEMENTATION DECISION]` Retain as decoded silence, with finite degeneracy conventions below. |
| Identifier parsing | `[IMPLEMENTATION DECISION]` Preserve strings, trim whitespace, reject duplicate identities and path collisions. |
| Vote counts | `[IMPLEMENTATION DECISION]` Require finite nonnegative integers; malformed rows fail the audit for correction. |
| Filename/layout | `[IMPLEMENTATION DECISION]` User-provided concatenation by default. Alternate confirmed templates must be explicit. |

Category counts include overlapping clips. Overlap count means the number of rows
with more than one positive stutter category; label-combination frequencies are also
saved. Missing, empty, invalid and short counts are disjoint statuses. The complete
audit includes nonselected clips, and task summaries separately describe selection.

## Preprocessing and framing

`[PAPER]` Resample to 8000 Hz, use 20 ms rectangular non-overlapping LTAS frames,
and retain full-band audio with the filtered components.

`[IMPLEMENTATION DECISION]` Mono is the arithmetic mean of channels. `soundfile`
decodes float64 audio; `scipy.signal.resample_poly` uses the rate ratio reduced by
GCD, a Kaiser anti-aliasing kernel with beta 5, and constant boundary extension.
The resampler's FIR design window is not a frame analysis window. Audio amplitude
is not peak-normalized, pre-emphasized or de-noised in the LTAS paths. The reference
MATLAB functions normalize by `max(x)`; the target paper does not specify that step,
so it is not silently inherited. This can affect unnormalized standard deviation/range.

For resampled length N, `T = floor(N / 160)` frames are formed from the first `160*T`
samples. The remaining `N mod 160` samples are reported as a discarded framing tail.
Filtering preserves N; full-component RMS uses all N samples. This consistent choice
differs from MATLAB `buffer` padding the final frame. Partial-frame policy is
**NOT SPECIFIED IN PAPER**. No empty input is padded into a fake feature vector.

For possibly complex component y_i and frame t:

```text
r_i[t] = sqrt( sum_{n=0..159} |y_i[160*t+n]|^2 / 160 )
R_i    = sqrt( sum_{n=0..N-1} |y_i[n]|^2 / N )
```

Full-band is component i=0. Squared complex magnitude is necessary for complex
bandpass output; taking only the real part would change energy.

## CQT: reconcile the count without silently changing B

`[PAPER]` Equations (4) and (5) give

```text
f_k = 10 * 2^(k/12)
b_k = f_k * (2^(1/12) - 1)
```

The valid integer indices below 4000 Hz are **k=0,...,103**: 104 logarithmic centers.
The paper's 106-filter dimension and its phrase “evenly spaced in frequency” do not
follow from those equations. Uniform linear spacing is not constant-Q spacing.

`[REFERENCE]` The LTFAT `cqtfilters` implementation linked through [21] surrounds
the logarithmic bank with a DC low-pass channel and a Nyquist high-pass channel.
Thus **104 interior + 2 boundary filters = 106**, and adding original audio gives
107 components. This is a defensible resolution, **not proof that the 2025 authors
used those boundary channels**. The reference MATLAB wrapper actually sets fmin=5;
this project deliberately uses the target paper's fmin=10 instead.

`[REFERENCE]` Interior prototype support in LTFAT is
`w_k = f_k*(2^(1/12)-2^(-1/12))`, the spacing between adjacent neighbor centers.
This is not equation (5)'s one-sided `b_k`. Both are documented rather than conflated.
Boundary supports are 20 Hz at DC and `2*(4000-f_103)` at Nyquist. Each support has a
minimum of four FFT bins. Fractional-bank peak gains are
`sqrt(N/ceil(N*w_i/8000))`, with the two endpoint gains divided by sqrt(2).
Boundary responses have a flat middle when their support exceeds the adjacent band.

`[IMPLEMENTATION DECISION]` For a clip of length N, the Python implementation samples
continuous raised-cosine frequency responses on its N-point DFT grid. It uses wrapped
distance d to center, width w and taper fraction q:

```text
flat = w*(1-q)/2
u = clip((d-flat)/(w*q/2), 0, 1)
H_i = gain_i * (0.5 + 0.5*cos(pi*u)); H_i=0 for d >= w/2
```

Interior q=1. Endpoint q is `min(1, adjacent_support/own_support)`.
`h_i = IFFT(H_i)` and `y_i = IFFT(FFT(x)*H_i)` implement **N-point circular
convolution**, the periodic boundary interpretation of a DFT filter bank. There is
no downsampling of band outputs. This gives N samples per filter while processing
one filter at a time. The paper does not specify edge conditions.

The raised cosine is a **frequency-domain filter shape** inherited from the
reference approach. Every LTAS **time frame remains rectangular**, never Hann or
Hamming. LTFAT uses rounded finite support vectors with sub-bin shifting/taper
details; continuous-grid sampling here is not exactly that numerical implementation.
`[NOT IMPLEMENTED]` Exact MATLAB parity, unknown original impulse responses, and
recovery of the authors' precise CQT construction.

## Ten LTAS measures: exact equations used

`[PAPER]` Names and `(M+1)*10-1` dimension are given in II-A. Their individual
equations and the deleted index are **NOT SPECIFIED IN PAPER**.

`[REFERENCE]` The linked `LTAS_feat_CQT.m` supplies the following statistic-major
order, which this project follows. Let `mu_i=mean_t r_i[t]`,
`s_i=std_t(r_i, ddof=1)` and `d_i=max_t r_i-min_t r_i`.

| Stored block | Formula | Components |
|---|---|---|
| 1. Component RMS normalized by full-band RMS | R_i / R_0 | i=1..M |
| 2. Normalized mean | mu_i / R_0 | i=0..M |
| 3. Standard deviation | s_i | i=0..M |
| 4. Frame RMS standard deviation / full-band RMS | s_i / R_0 | i=0..M |
| 5. Frame RMS standard deviation / band RMS | s_i / R_i | i=0..M |
| 6. Skewness | mean((r_i-mu_i)^3) / mean((r_i-mu_i)^2)^(3/2) | i=0..M |
| 7. Pearson kurtosis | mean((r_i-mu_i)^4) / mean((r_i-mu_i)^2)^2 | i=0..M |
| 8. Range | d_i | i=0..M |
| 9. Normalized range | d_i / R_0 | i=0..M |
| 10. Pairwise variability | mean_t abs(r_i[t+1]-r_i[t]) / R_0 | i=0..M |

The word “frame standard deviation” is interpreted by the reference code as the
standard deviation of **frame RMS values across time**, not within-frame waveform
standard deviation. The reference code uses absolute successive differences; its
supporting PDF's typeset expression omits the absolute value. We follow the code.

Block 1 omits i=0: R_0/R_0 is redundant for nonzero full-band RMS. No arbitrary
last column is removed. There are M + 9*(M+1) = 10*(M+1)-1 features.
For CQT, `feature_1..feature_106` are block 1, then nine blocks of 107 values.
Full-band precedes filters inside each subsequent block. This order is saved in the
model configuration. It differs from the order of names in the paper, not the set
of measures.

`[REFERENCE]` Sample standard deviation (`ddof=1`), biased population skewness and
Pearson rather than excess kurtosis follow the MATLAB function defaults.
`[IMPLEMENTATION DECISION]` A single frame has std=0 and pairwise variability=0.
Constant or numerically degenerate frame-RMS sequences have skewness=kurtosis=0
(conventions, not their mathematically undefined values). A denominator no larger
than float64's smallest normal positive value produces 0. No blanket NaN-to-zero
replacement is used: unexpected non-finite final features are rejected and logged.

## Other feature paths

### GT-LTAS

`[PAPER]` 32 filters, 0–4000 Hz, original signal included, 329 features, and
`g(t)=a*t^(N-1)*exp(-2*pi*b*t)*cos(2*pi*fc*t+phi)` with
`b=24.7*(4.37*fc_kHz+1)`.

`[IMPLEMENTATION DECISION]` Order 4 (paper gives general range 3–5), phase 0,
32 equally spaced points on log(1+0.00437*f) including 0 and 4000, truncated impulse
at `2*pi*b*t=30`, normalized to unit response at center. Convolution is causal,
zero initial state, and cropped to the first N samples. These numerical choices are
**NOT SPECIFIED IN PAPER**. No factor 1.019 is silently added to ERB bandwidth.
The reference wrapper selects voiced samples; that is not specified in the primary
paper, so this implementation does not discard unvoiced speech.

### SFF-LTAS

`[PAPER]` Equation (6) is implemented directly with first-order complex recursion:
`y[n]=x[n]+a_k*y[n-1]`, `a_k=.98*exp(-j*2*pi*f_k/8000)`,
`f_k=0,20,...,4000`. There are 201 filters + full-band, giving 2019 dimensions.
`[IMPLEMENTATION DECISION]` Zero initial state, causal N-sample output, squared
magnitude RMS. No unrelated gain adjustment or pre-emphasis is added.

### Plain LTAS baseline

`[NOT IMPLEMENTED]` A verified reconstruction of the primary paper's unspecified
plain LTAS baseline. Its filter bank and dimension cannot be identified from its text.

`[REFERENCE]` The executable `LTAS` option is a clearly labeled **proxy** using the
linked `LTAS_feat.m` 8 kHz branch: 9 FIR responses, edges
0,30,60,120,240,480,960,1920,2400,3840 Hz, order 32, frequency-sampling design,
and full-band, giving 99 values. Its exact design grids are in `banks.py`.
`[IMPLEMENTATION DECISION]` SciPy `firwin2` with its default Hamming design taper and
centered convolution cropping is used instead of MATLAB `fir2`. This taper designs
the filter kernel; it is not applied to the 20 ms frames. This baseline's scores
cannot establish reproduction of the table's plain LTAS row.

### MFCC

`[PAPER]` 13 static + 13 delta + 13 double-delta, followed by mean, std, kurtosis,
skewness across frames: 156 dimensions.

`[IMPLEMENTATION DECISION]` Rectangular 20 ms frames, hop 20 ms, no overlap, FFT 256
(zero extension for the transform only), 26 Slaney-normalized Mel filters over
0–4000 Hz, power spectrum, power-to-dB floor 1e-10 without top-dB clipping,
orthonormal DCT-II, c0 included, no lifter or pre-emphasis. These settings are
**NOT SPECIFIED IN PAPER**. The 25 ms Hamming/10 ms hop setting in [21] is not silently
adopted because the user's framing instructions specify rectangular non-overlap.

### ZTWCC

`[PAPER]` nFFT=128 is the reported configuration; dimension is the same 39*4=156.
The paper describes a decaying window, numerator of group delay, second differences,
Hilbert envelope and cepstrum. Other tested nFFT values are not used here.

`[REFERENCE 35]` Use a 5 ms segment, 6.25 ms hop, multiply by w1^2*w2, with
`w1[0]=0`, `w1[n]=1/(4*sin(pi*n/(2*nFFT))^2)` and
`w2[n]=4*cos(pi*n/(2*M))^2`. For the windowed segment z, compute FFT(z) and
FFT(n*z), take the real part of their conjugate product (NGD), then twice-difference
over frequency, Hilbert-envelope and real IFFT of the log spectrum.

`[IMPLEMENTATION DECISION]` Pre-emphasis coefficient .97, centered circular second
difference, natural logarithm floor 1e-12, first 13 coefficients including c0,
and discard incomplete final segments. Reference [35] instead uses nFFT=2048 and
14 coefficients; primary-paper values 128 and 13 take precedence. Exact adaptation
details in the stutter experiment are **NOT SPECIFIED IN PAPER**.

For **both** cepstral paths, `[IMPLEMENTATION DECISION]` delta uses the +/-2-frame
regression formula `sum_{n=1..2} n*(c[t+n]-c[t-n])/10` with edge replication.
Double delta applies that operator again. Statistics are stored as four consecutive
39-value blocks: mean, sample std, Pearson kurtosis, skewness; degeneracies follow
the LTAS conventions. ZTW's specialized window is its reported transform, not an
overlapping LTAS framing stage.

## Training, probabilities and evaluation

| Item | Provenance |
|---|---|
| One clean/target binary task per stutter type | `[PAPER]` |
| Split ratio .33 | `[PAPER]`; interpreting it as test_size=.33 is `[IMPLEMENTATION DECISION]`. |
| Stratified random clip split, seed 42 | `[IMPLEMENTATION DECISION]` NOT SPECIFIED IN PAPER. Same seed reused across classifiers. |
| StandardScaler | `[IMPLEMENTATION DECISION]` NOT SPECIFIED IN PAPER; fitted only to original training rows; optional `none`. |
| SMOTE | `[PAPER]`; training-only after split/scaling is `[IMPLEMENTATION DECISION]` requested to prevent leakage. |
| SMOTE k=5, random_state=42, sampling_strategy=auto | `[IMPLEMENTATION DECISION]` Standard defaults; too few samples raises an error. |
| SVM RBF | `[PAPER]` |
| SVM C=1, gamma=scale, other SVC defaults | `[IMPLEMENTATION DECISION]` NOT SPECIFIED IN PAPER; no tuning. |
| SVC probability=True | `[IMPLEMENTATION DECISION]` Internal training-only probability calibration for the requested demo. F1 uses SVC.predict. |
| BCE, sigmoid, Adam, learning rate .01 | `[PAPER]`; fused BCEWithLogitsLoss is the stable equivalent. |
| Neural input shape (batch,1,D) | `[IMPLEMENTATION DECISION]` Author input arrangement NOT SPECIFIED IN PAPER. No invented time sequence is claimed. |
| Layers=1, hidden=32, dropout=0, batch=32, epochs=20 | `[IMPLEMENTATION DECISION]` All NOT SPECIFIED IN PAPER, configurable. |
| Adam betas=(.9,.999), eps=1e-8, weight_decay=0 | `[IMPLEMENTATION DECISION]` PyTorch defaults, NOT SPECIFIED IN PAPER. |
| Batch shuffling, workers=0, float32 neural computation | `[IMPLEMENTATION DECISION]` Repeatable simple loader and small-memory training. |
| Last hidden state; forward/reverse concatenation in Bi-LSTM; linear output | `[IMPLEMENTATION DECISION]` NOT SPECIFIED IN PAPER. |
| Neural class threshold .5 | `[IMPLEMENTATION DECISION]` No threshold tuning. |
| F1 score | `[PAPER]`; positive-class binary averaging and zero_division=0 are `[IMPLEMENTATION DECISION]`. |
| Fixed-epoch training; no early stopping | `[IMPLEMENTATION DECISION]` No unreported validation/tuning procedure added. |
| FluencyBank independent test | `[PAPER]`; reuse all fitted objects, do not train or balance external data. |
| GPU selection/fallback, saving formats, local demo | `[IMPLEMENTATION DECISION]` Software requirements, not scientific methodology. |

The single-step recurrent input has no multi-frame temporal sequence; it is a
minimal functional architecture for aggregated feature vectors. Bidirectionality
still has two learned directions but cannot recover unprovided temporal structure.
**Its reported F1 must not be attributed to the paper's unknown LSTM architecture.**

The paper's prose mentions balancing before it describes splitting, but does not
provide executable ordering details. This implementation never lets SMOTE see test
rows. An automated perturbation test changes the test features by a huge offset and
verifies that original-training scaling and synthetic training samples remain identical.
Internal SVM probability fitting uses training data only; it is not evaluated as an
independent calibration study. SMOTE also changes the effective training class prior.

## Reproducibility and limitations

`[IMPLEMENTATION DECISION]` Seeds are set for Python, NumPy, PyTorch and SMOTE/split.
cuDNN benchmarking is off and deterministic cuDNN mode is on. Bitwise agreement
across GPUs, devices, package versions or operating systems is not guaranteed.
Every completed run records source hashes, feature-file/selection hashes, package
versions, seed, parameters, actual device, counts and the explicit split IDs.

Feature CSVs are checked against their saved hashes and extraction configuration.
Model loading rejects mismatched feature settings. The trained scaler travels with
the model and is reused by both CLI and browser inference. FluencyBank identities
are checked against the original training identities; this is not an acoustic
near-duplicate or speaker-identity audit.

`[NOT IMPLEMENTED]` Recovery of missing author parameters, proof of MATLAB feature
equality, speaker-disjoint guarantees, or actual SEP-28k/FluencyBank reproduction
without access to those recordings. No paper score is substituted for a measured
score. The supplied software test report is separate from real-project validation.
Windows CPU/GPU execution and real model usefulness must be verified on the user's
Windows PC. No extra architectures, augmentation or parameter sweeps are introduced.
