# Enhancing Stutter Detection using Long-Term Average Spectrum Values

A Windows-ready, beginner-oriented Python implementation of the pipeline in the
ICASSP 2025 paper by Narasinga et al.
[Paper DOI: 10.1109/ICASSP49660.2025.10888625](https://doi.org/10.1109/ICASSP49660.2025.10888625).

**This is a reference-informed reproduction with explicit implementation decisions,
not a claim of identical author code or reproduced paper scores.** Read
[METHODOLOGY.md](METHODOLOGY.md) before interpreting a result. It distinguishes
`[PAPER]`, `[REFERENCE]`, `[IMPLEMENTATION DECISION]`, and `[NOT IMPLEMENTED]`.

The supplied dataset is on your Windows PC. This package contains functioning code
and software-test evidence. Real dataset CSVs, measured F1 scores and trained research
models are generated when you run the commands below on that PC. They are not
fabricated or replaced by models trained on test tones.

## 1. What the project does

1. Inspect every SEP-28k label row and decode every corresponding WAV.
2. Select clean versus one **pure** target stutter type.
3. Preserve the original selection and save a separate valid selection.
4. Convert audio to mono and resample to 8 kHz.
5. Run the CQT filter bank; retain the original full-band signal.
6. Use 20 ms rectangular, non-overlapping frames and calculate RMS.
7. Calculate ten LTAS statistics and remove the redundant full-band ratio.
8. Save a 1,069-dimensional feature vector with each clip's identity.
9. Split into training/test sets; fit scaling and SMOTE using training data only.
10. Train SVM, LSTM or Bi-LSTM and evaluate untouched test data using F1.
11. Save the model, exact settings, split IDs, reports, confusion matrix and curves.
12. Compare measured results against separately stored paper results.
13. Optionally evaluate the fixed model on FluencyBank and run a local upload demo.

The other implemented feature paths are GT-LTAS (329), SFF-LTAS (2019), MFCC (156),
ZTWCC (156 with nFFT=128), and a 99-dimensional reference-based LTAS baseline proxy.
The identity of the paper's plain LTAS baseline is **NOT SPECIFIED IN PAPER**.

## 2. Folder structure

Extract the project folder so that this file is `D:\stutter-detection\README.md`.

```text
D:\stutter-detection
├── dataset
│   ├── clips\stuttering-clips\clips\   actual WAV directory
│   ├── SEP-28k_labels.csv
│   ├── SEP-28k_episodes.csv
│   ├── fluencybank_labels.csv
│   └── fluencybank_episodes.csv
├── data                               selections and extracted features
├── models                             one folder per actual training run
├── results                            measured outputs and paper references
├── src                                small reusable Python modules
├── requirements.txt
├── README.md
└── METHODOLOGY.md
```

The program finds the project root from its own location. After copying to Windows,
the default paths are exactly the paths above. No Mac-specific paths are required.
An optional `--root` before the command allows a different project root.

## 3. Install Python 3.14 and create venv

Open the folder in VS Code. In its **PowerShell terminal**, run:

```powershell
cd D:\stutter-detection
py -3.14 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m src.pipeline setup
```

These commands explicitly use the virtual environment, so you do not need to change
PowerShell execution policy or activate it. In VS Code choose **Python: Select
Interpreter**, then select `D:\stutter-detection\venv\Scripts\python.exe`.

Python 3.14 is intentional. Dependency pins were exercised with Python 3.14.6 on
macOS; Windows wheel availability is separately recorded in the results folder when
verified. An installed Windows runtime and GPU still require local validation.

### Optional RTX 3050 GPU

Signal processing and SVM use the CPU. Only LSTM/Bi-LSTM may use CUDA.
Use the [official PyTorch installer](https://pytorch.org/get-started/locally/) to select
Windows, Pip and a CUDA build supported by your NVIDIA driver. Keep the tested Torch
version where that build is available; record any intentional change with `pip freeze`.
Do not install torchvision or torchaudio: this project does not use them.

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
.\venv\Scripts\python.exe -m pip freeze > results\installed_packages_windows.txt
```

The default `--device auto` uses CUDA when available and otherwise CPU. A CUDA memory
or backend failure is logged, and the run restarts from the same seed on CPU.
`--device cuda` fails explicitly if CUDA cannot be used. `--device cpu` is always
available. Start with batch size 32 for the 4 GB card; lower it explicitly if needed.

## 4. Dataset setup and first audit

Place the existing dataset in `D:\stutter-detection\dataset` as shown above. The code
does not download podcasts, modify recordings or use the episodes CSVs for training.
Those CSVs remain available as dataset metadata.

The default WAV name is exactly `{Show}{EpId}{ClipId}.wav`, preserving CSV IDs as text.
There is no guessed filename fallback. Run:

```powershell
.\venv\Scripts\python.exe -m src.pipeline prepare --target block
```

This scans **all** label rows and WAVs, not only the chosen classes. It prints:
total/clean/category counts, overlap combinations, missing/empty/invalid/short audio,
and the original/valid counts for the selected task. It writes:

- `data\sep28k_audio_audit.csv`
- `data\clean_vs_block.csv` (original selection, including invalid recordings)
- `data\clean_vs_block_valid.csv` (only actual usable audio)
- `results\sep28k_dataset_summary.json`
- `results\processing_errors.csv`

If every file is missing, inspect the template and the real directory before continuing.
For an explicitly confirmed underscore naming convention, for example:

```powershell
.\venv\Scripts\python.exe -m src.pipeline prepare --target block --wav-template "{Show}_{EpId}_{ClipId}.wav"
```

For a confirmed nested convention, use `"{Show}/{EpId}/{Show}_{EpId}_{ClipId}.wav"`.
The chosen template is recorded. Changing filenames is never automatic.

### Label rules

Clean means all five counts (`Block`, `Interjection`, `SoundRep`, `WordRep`,
`Prolongation`) equal zero. Pure Block means Block > 0 and the other four equal zero.
The same pure-target rule is used for the other four tasks. Clean is 0; target is 1.

**Pure-label selection is our implementation choice, NOT SPECIFIED IN PAPER.**
Counts > 0 mean at least one positive annotation; we do not silently use majority
voting. We do not add filters on `Unsure`, `NoSpeech`, `Music`, quality flags or
`NoStutteredWords`. Consequently “clean” is the requested label rule, not proof of
fluent, high-quality speech. Overlapping clips are counted and excluded from pure tasks.

Zero-sample WAVs are rejected even when their headers look valid. Missing, corrupt,
non-finite and shorter-than-20-ms recordings receive distinct log entries. A decoded
silent recording with samples is different from an empty recording; its statistics
use documented zero-denominator conventions. No speech or features are fabricated.

## 5. Validate one clip, then extract

You may inspect a known valid WAV explicitly:

```powershell
.\venv\Scripts\python.exe -m src.pipeline validate-clip "D:\stutter-detection\dataset\clips\stuttering-clips\clips\YOUR_ACTUAL_CLIP.wav"
```

Replace `YOUR_ACTUAL_CLIP.wav` with a real filename. Extraction automatically validates
the first selected valid clip **before** opening the full feature output:

```powershell
.\venv\Scripts\python.exe -m src.pipeline extract --target block --feature CQT-LTAS
```

For a normal 3-second clip, expect 24,000 resampled samples, 160-sample frames,
150 frames, 106 filter outputs plus full-band, RMS shape `(107, 150)`, and vector
shape `(1069,)`. The report includes original rate/duration and NaN/Infinity counts.
Real durations are measured. Trailing incomplete frames are dropped and counted;
the waveform itself is not forced to 3 seconds or padded with artificial speech.

`data\ltas_features.csv` contains `Show,EpId,ClipId,label,feature_1,...,feature_1069`.
Adjacent JSON files record the validation, feature settings, failures and checksums.
Progress prints every 100 clips. Audio and filters are processed one at a time;
feature rows stream to disk. A `.partial.csv` remains if a process is interrupted;
rerunning extraction starts a fresh pass. Failed clips are logged and counted, never
silently treated as valid. Final validation flags incomplete extraction.

## 6. Train the models

```powershell
.\venv\Scripts\python.exe -m src.pipeline train --target block --classifier SVM
.\venv\Scripts\python.exe -m src.pipeline train --target block --classifier LSTM
.\venv\Scripts\python.exe -m src.pipeline train --target block --classifier Bi-LSTM
```

Or run the entire initial project in one command:

```powershell
.\venv\Scripts\python.exe -m src.pipeline run --target block --feature CQT-LTAS --classifier all
```

The order is **stratified split (33% test) -> fit training scaler -> SMOTE on training
features -> train -> evaluate untouched test**. Before/after SMOTE counts are printed.
Scaling is a documented implementation choice, optional with `--scaling none`.
The paper's prose places SMOTE earlier; it does not give enough detail to establish
its actual implementation order. We use the leakage-safe order requested here.

SVM uses the paper's RBF kernel. `C=1` and `gamma=scale` are standard implementation
choices, not paper values. No hyperparameter search is performed.

LSTM and Bi-LSTM use sigmoid probabilities, binary cross-entropy and Adam at 0.01.
The actual loss is `BCEWithLogitsLoss`, the numerically stable combined sigmoid+BCE.
The architecture is a configurable simple default: one recurrent layer, hidden size
32, no dropout, batch size 32, 20 epochs. **All those architecture/training-size details
are NOT SPECIFIED IN PAPER.** The aggregate feature vector is represented as one
timestep; this makes no claim about the authors' undisclosed temporal arrangement.
Bi-LSTM has forward and reverse recurrent directions. No test-set tuning, validation
early stopping or architecture search is used. Curves show training loss only.

Example explicit configuration:

```powershell
.\venv\Scripts\python.exe -m src.pipeline train --classifier LSTM --hidden-size 32 --layers 1 --epochs 20 --batch-size 32 --seed 42 --device auto
```

If SMOTE has fewer than six minority training examples, it fails with the counts.
Choose `--smote-k` explicitly for a small legitimate dataset; it is never reduced silently.

## 7. All five experiments and optional feature types

```powershell
.\venv\Scripts\python.exe -m src.pipeline run --target all --feature CQT-LTAS --classifier all
```

Targets are `block`, `interjection`, `soundrep`, `wordrep`, `prolongation`.
This produces 15 binary experiments. It does not produce a multiclass classifier.

For a separate feature path:

```powershell
.\venv\Scripts\python.exe -m src.pipeline run --target all --feature GT-LTAS --classifier all
```

Valid feature names: `CQT-LTAS`, `GT-LTAS`, `SFF-LTAS`, `LTAS`, `MFCC`, `ZTWCC`.
`--feature all` runs all six paths (90 runs with all targets/classifiers), including
the explicitly qualified LTAS baseline proxy. Start with CQT-LTAS. No exact runtime
is promised for the full dataset; it depends on CPU, disk, GPU and clip durations.

## 8. Results and comparison

Each run gets a timestamped folder under `models` and `results`. Look for:

- `model.joblib`: trained estimator/network weights and the fitted scaler.
- `configuration.json`: parameters, versions, device, data counts, hashes and SMOTE counts.
- `split.csv`: original training/test clip identities.
- `metrics.json`, `classification_report.txt`, `predictions.csv`.
- `confusion_matrix.csv`, `confusion_matrix.png`.
- For neural models: `training_history.csv`, `training_curve.png`.

`results\results.csv` contains exactly `stutter_type,classifier,feature_type,f1_score`.
Repeated experiments append rows; their timestamped folders and the run list in
`results\experiment_config.json` identify the runs. Paper scores live only in
`results\paper_reference_results.csv` (90 SEP-28k rows + 30 FluencyBank rows, including
all GT-LTAS values). Comparison preserves separate measured and paper F1 columns:

```powershell
.\venv\Scripts\python.exe -m src.pipeline compare
```

F1 here is **binary F1 for label 1**. The paper does not disclose the averaging
convention. Differences in scores cannot by themselves prove an implementation bug
or prove faithful reproduction. Model probabilities are estimates; SVM decision
labels and calibrated probability argmax can occasionally disagree.

## 9. FluencyBank independent evaluation

FluencyBank is never accepted as a training dataset. Prepare/extract it separately:

```powershell
.\venv\Scripts\python.exe -m src.pipeline prepare --dataset fluencybank --target all
.\venv\Scripts\python.exe -m src.pipeline extract --dataset fluencybank --target all --feature CQT-LTAS
.\venv\Scripts\python.exe -m src.pipeline evaluate --model "models\YOUR_SEP_RUN_FOLDER\model.joblib"
.\venv\Scripts\python.exe -m src.pipeline compare --dataset fluencybank
```

Replace the model path with one printed by training. Evaluation infers its target and
feature type from the model. For GT-LTAS, extract FluencyBank GT-LTAS and load a saved
SEP-28k GT-LTAS model. `--wav-dir` and `--wav-template` on `prepare` can select a
different confirmed FluencyBank layout. Evaluation applies the existing scaler and
weights; no SMOTE or fitting takes place. Clip-ID overlap with training is rejected.

## 10. Local Clean vs Block WAV demo

After training a CQT-LTAS Clean vs Block model:

```powershell
.\venv\Scripts\python.exe -m src.pipeline demo --model "models\YOUR_BLOCK_CQT_RUN_FOLDER\model.joblib"
```

Open **http://127.0.0.1:8501**. Choose a WAV and press **Analyze audio**. It displays
the duration, original and analysis sample rates, Clean/Block prediction, estimated
probabilities and feature dimension. The recording stays on your computer. The
server binds only to localhost, accepts files up to 25 MB and removes each temporary
upload after processing. Press Ctrl+C to stop it. No extra web-framework package is needed.

For command-line inference:

```powershell
.\venv\Scripts\python.exe -m src.pipeline predict "D:\path\clip.wav" --model "models\YOUR_BLOCK_CQT_RUN_FOLDER\model.joblib"
```

Only load trusted `joblib` files made by this project. Both interfaces call the same
`extract()` and `predict_features()` used in model verification.

## 11. Automated validation and troubleshooting

```powershell
.\venv\Scripts\python.exe -m src.pipeline selftest
.\venv\Scripts\python.exe -m src.pipeline validate
```

`selftest` uses explicitly synthetic numerical fixtures in an isolated temporary
directory. It tests missing, empty, invalid and short WAVs; resampling/framing;
statistics against hand-calculated values; all feature dimensions; overlapping labels;
test-set leakage protection; every binary task; all three classifiers; model saving;
external evaluation; and actual HTTP WAV uploads. Test audio and models are removed.
The resulting software report/log are clearly marked **not research results**.

`validate` checks your real project. Exit code 0 means its listed checks passed;
2 means real work is pending or failed; 1 indicates a command error. Without the
Windows dataset, it must not report complete. A package built on macOS cannot certify
your Windows CUDA driver or a real-data result.

If labels/files are absent, check the paths. If dependencies fail, confirm Python 3.14,
the selected venv, network access and `pip check`. If an edited feature CSV fails its
checksum, rerun extraction. Do not remove the guard. Inspect
`results\processing_errors.csv` for audio and CUDA errors.

## 12. Limitations that matter

- The paper does not fully specify its labeling, filter implementation, architecture,
  normalization, split seed, F1 convention or all cepstral parameters.
- CQT is a Python reference-informed construction, not numerically verified against MATLAB.
- Its 106 outputs include DC/Nyquist fillers inferred from the cited library. This
  reconciles the count without changing the 12-bin logarithmic interior spacing.
- The split is clip-level, as a minimal interpretation. Speakers or episodes may occur
  in both sets; speaker-independent performance is not established.
- A one-timestep recurrent architecture is a necessary minimal choice, not recovered
  author architecture. The plain LTAS baseline is a qualified reference proxy.
- Numerical tests establish software behavior, not useful stutter recognition.
- Empty recordings are excluded, not replaced. The valid count may differ from 9,340.
- This demo is a research interface for a binary label rule, not a diagnostic system.

See [METHODOLOGY.md](METHODOLOGY.md) for formulas, provenance, source differences and
every material default. No augmentation, additional classifiers or model sweeps are included.
