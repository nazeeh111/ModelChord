# Reproduce v0.2

Use a fresh output directory. The released confirmation has been observed; replaying it is reproduction, not new untouched research evidence. Do not edit frozen artifacts, split roles or recipes in place.

Install the pinned package and study/development dependencies using the [README](../README.md#run-locally). `requirements-macos.lock` records the tested environment; Linux installs the CPU PyTorch wheel first. The recorded numerical environment is Python 3.12.13, PyTorch 2.9.1, NumPy 2.2.6, SciPy 1.16.3, scikit-learn 1.7.2 and safetensors 0.6.2, CPU / two threads. Exact weight bytes can vary across platforms or math libraries.

## Main-suite replay

```bash
python scripts/reproduce_v2.py --output /tmp/modelchord-v02-new-replay
```

The [wrapper](../scripts/reproduce_v2.py) refuses an existing destination. It copies source and the unchanged old study/parent artifacts, downloads the official free optical, YouTube and SMS archives, verifies recorded archive/member hashes, reconstructs text calibration/selection arrays from frozen source IDs and checks exact equality, runs all 12 v0.2 main recipes, then freezes the replay. It does not retrain old experts or reproduce every historical pilot decision. Network is needed for these public data downloads, not model inference. Raw text stays in ignored replay folders and is not a publication asset.

[reproduction-check.json](../experiments/v02/reproduction-check.json) records an executed complete fresh-workspace replay: all 12 weight files exactly match the frozen selected recipe outputs, official archive hashes passed, and new confirmation was not replayed. Timing and report metadata need not be byte-identical. The copied protocol describes the historical experiment; a replay's new lock/time is not evidence of a new reserved cohort.

For preparation only:

```bash
python scripts/reproduce_v2.py --output /tmp/modelchord-v02-preparation --prepare-only
```

To deliberately replay the already public confirmation in a new workspace:

```bash
python scripts/reproduce_v2.py --output /tmp/modelchord-v02-confirmation-replay --replay-confirmation
```

That flag evaluates all declared participants after freezing. It must not be used to select new recipes or described as an untouched test. The evaluator refuses an existing confirmation marker/result; do not remove one to bypass its one-shot guard. A revised method requires genuinely new reserved source evidence.

## Source-role and integrity checks

With the official SMS source available at `study/private/sms.zip`, independently regenerate only development arrays into a fresh file:

```bash
python scripts/prepare_v2_development.py --output /tmp/modelchord-text-development-check.npz --check
```

This [preparation command](../scripts/prepare_v2_development.py) uses frozen calibration and selection source-ID lists, excludes old confirmation, checks all 16 arrays and refuses an existing output. It performs no model inference. The old vision arrays are bound at `study/workbench.npz`; their generation is retained in `scripts/run_study.py`.

In a complete local study workspace, including exact raw source files:

```bash
python experiments/v02/evaluate_locked.py verify
python -m pytest -q tests experiments/v02/test_search.py
```

Verification checks all 272 v0.2 bound paths and the old 327-path historical lock with the explicit initializer version exception. The archived original initializer retains exact old bytes. A clean source clone excludes raw SMS/YouTube files; use the preparation/replay route to retrieve hash-checked sources before raw-file verification. Do not regenerate source ledgers over the frozen originals.

The [independent review](../experiments/v02/review/FINAL_REVIEW.md) verifies exact acceptance decisions, data roles, counters, rank bounds and immutable bytes. The 19 synthetic search tests are software checks, not confirmation inference. Existing conflict/insufficient-calibration/easy controls and the negative v0.1 study remain historical evidence.

## Build profiling and installation checks

Run resource repeats in the fresh replay workspace, preserving the released measurements:

```bash
cd /tmp/modelchord-v02-new-replay
python scripts/measure_v2_build.py
python scripts/measure_v2_runtime.py --output /tmp/modelchord-v02-runtime-repeat.json
python -m build
python scripts/ci_smoke.py
```

The build profiler repeats the fixed vision11 repair/joint/fixed-factor recipes in three separate processes and writes the replay's derived `experiments/v02/build-profile.json` (overwriting that derived path on subsequent runs). It requires exact frozen weight matches and performs no new-cohort inference. Profile numbers are noisy and machine-specific. It includes data/parent loading, selection, complete search and static export; process RSS includes imports. Expert training and inherited RegMean construction are excluded. This is not an exact-FLOP comparison.

The runtime command measures both fixed deployment artifacts on synthetic inputs, retaining batch-one loaded-forward, preprocessing and safe per-call loading timings at the requested fresh output. It refuses an existing output and performs no cohort inference. Read its warm-cache, thread, startup/transport exclusion and shared process-memory scope before comparing numbers. [Recorded latency](RESULTS_V02.md#static-inference-latency).

The smoke command requires the package installed. It copies only each locked deployment artifact into a separate temporary working directory, starts Python with `-I` and asserts changed inputs alter logits. It does not itself enforce OS-level parent/cache/network denial.

For the stronger macOS test, install the built wheel into a separate environment outside the project, then run once per selected artifact:

```bash
python scripts/verify_independent.py --python /tmp/modelchord-clean-env/bin/python --artifact experiments/v02/main/vision-seed11-repair/model --deny "$PWD" --output /tmp/modelchord-v02-independent-vision.json
python scripts/verify_independent.py --python /tmp/modelchord-clean-env/bin/python --artifact experiments/v02/main/text-seed43-repair/model --deny "$PWD" --output /tmp/modelchord-v02-independent-text.json
```

The runner copies only the selected artifact and disables user-site/PYTHONPATH influence. macOS `sandbox-exec` denies project/parent reads, common caches and all network, and the worker checks denial and changed outputs. Linux needs an equivalently isolated networkless environment for the same claim. These are execution instructions; a passed reproduction or local test does not imply remote CI, download or release checks passed. Inspect actual delivery evidence separately.

Dataset attribution and transformations are recorded in the [source manifest](../experiments/v02/data/source-manifest.json), [source audit](../experiments/v02/review/DATA.md) and [results](RESULTS_V02.md). Preserve UCI CC BY 4.0 credit; do not publish raw SMS or YouTube comments.
