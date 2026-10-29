# Reproduction and installation

The repository includes the frozen study artifacts. Do not run a new training phase in place; exports are immutable and the confirmation set has already been observed.

Run the README quickstart to install the pinned package, tests and study dependencies. `requirements-macos.lock` records the complete tested macOS development environment; Linux CI installs the CPU PyTorch wheel separately. There is no paid API, key or external inference service.

```bash
python scripts/reproduce.py --output /tmp/modelchord-new-study
```

This creates a fresh directory, copies only code/protocol source and runs expert training, all registered baseline/repair/control recipes and a new lock. It downloads the official free UCI SMS source when absent. It does not copy raw private data or overwrite the released study. Installed study dependencies are required. `--confirm` also evaluates the declared confirmation after freezing. A deterministic reproduction uses the same publicly declared source partition and is **not** a new untouched research confirmation once you have seen the released results. A revised method or new tuning requires genuinely new reserved source data, not merely rerunning this split.

The final recipe uses160 epochs for clean digit bases,65 for shifted specialists and1,200 at the amended lower learning rate for inverted specialists. Two vision seeds and a bounded rank 4 text-adapter classifier are small CPU experiments. Exact float bytes may differ across PyTorch/BLAS/platform versions; source, data and recipe identities remain recorded. The local measured repeat reproduced the selected arithmetic-repair weights exactly on the tested machine.

The initial65-epoch inversion pilot and its diagnostic checkpoints remain in `study/pilot-v1` and the global recipe registry. The reproduction wrapper executes the final main recipe, not every historical pilot decision. RegMean/TSV and all required main controls/ablations actually execute. `--phase confirm` refuses a missing lock, existing results or prior one-shot marker. Original source IDs, processed content, all model manifests/weights, tokenizer/feature source and recipe hashes are checked before evaluating confirmation.

```bash
python -m build
python scripts/ci_smoke.py
```

The wheel contains the typed API, CLI and local web assets. Artifacts/datasets are distributed separately in the source repository and release ZIPs. Model inference itself does not download anything. For macOS strict independent verification, install the wheel into a separate venv outside this repository and run:

```bash
python scripts/verify_independent.py --python /tmp/clean-env/bin/python \
  --artifact artifacts/vision-seed11-v2/repair-strongest --deny "$PWD" \
  --output /tmp/independent-proof.json
```

That runner copies only the artifact and starts the installed package with user-site/PYTHONPATH disabled. The OS sandbox denies this whole project, the usual cache directories and all network. It asserts changed inputs alter logits and verifies attempted parent/network access fails. `sandbox-exec` is macOS-specific; on Linux use an equivalently isolated networkless container. The ordinary CI smoke is narrower and does not claim OS-level network denial. Initial timeout and missing-dependency attempts were retained before the passing installed runs.
