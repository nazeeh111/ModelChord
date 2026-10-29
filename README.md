# MODELCHORD

Compose compatible specialists into **one static model**, measure capability shortfalls, and attempt a bounded low-rank repair. The exported artifact runs with its parent experts unavailable.

**Experimental negative-result release.** Repair reduced the worst vision-task deficit on two seeds, but no composed candidate passed every frozen selection requirement. Text passed all confirmation floors after missing its selection gate, so its acceptance status also remains negative. MODELCHORD returns the measured frontier and `no acceptable candidate found`. It makes no general superiority or universal capability guarantee.

![Local composition workbench](docs/media/workbench-final-floors.png)

## Run locally

Python 3.11–3.13. The tested environment uses Python 3.12, pinned PyTorch 2.9.1, NumPy 2.2.6 and safe tensors. No inference API, key or paid service is needed.

```bash
git clone https://github.com/nazeeh111/ModelChord.git
cd ModelChord
python3.12 -m venv .venv
source .venv/bin/activate
# On Linux, install the CPU wheel first to avoid unnecessary CUDA dependencies:
# python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[study,dev]'
python -m pytest -q
modelchord serve --project . --port 8765
```

Open [the local workbench](http://127.0.0.1:8765). Choose specialists, floors, correction rank and rounds; inspect results; export safe weights and the recorded recipe. The server binds only to loopback. Its build checks use permitted calibration/selection data, never the confirmation partition. See [the owner walkthrough](docs/WALKTHROUGH.md).

Run the static artifacts in separate applications:

```bash
python examples/predict.py artifacts/vision-seed11-v2/repair-strongest --input study/demo-vision-inputs.json
python examples/text_app.py artifacts/text/repaired 'Meeting at two.' 'WIN a FREE prize!'
```

Each application loads only its exported artifact. The vision head lists ten digits; text lists `ham` and `spam`. This is a small classifier and text-adapter integration, not an arbitrary architecture merger or a generative language model. Public artifacts and the installable wheel are also attached to [the versioned release](https://github.com/nazeeh111/ModelChord/releases/tag/v0.1.0). No PyPI publication is implied.

## What the builder does

- Validates the actual common-base identity, architecture, tensor names, precision, labels, preprocessing, tokenizer and source-license policy. Matching shapes alone are insufficient. Unknown architectures and tied tensors are rejected by this bounded v0.1 registry.
- Exports task arithmetic, weight averaging, TIES/DARE-style, TSV spectral and activation-informed RegMean baselines, using layerwise source reads.
- Measures per-task accuracy deficits and protected margins. Calibration gradients propose layer/subspace edits; actual selection interventions allocate a limited correction-rank budget. Accepted corrections are folded into static weights; rejected steps roll back.
- Optimizes worst deficit before aggregate deficit and loss. It avoids normalizing by unstable expert-minus-base gains, retains a zero-correction fast path, and records unsuccessful searches.
- Keeps source artifacts immutable, loads numeric safe-tensor files without pickle or remote code, and binds recipes to model, preprocessing and evaluation/data identities.

Repair uses labeled data and gradient optimization. It is **data-assisted**, not training-free. Ordinary full-weight joint fine-tuning is an explicitly separate baseline. The typed [API/CLI](docs/API.md) and [evaluator extension](examples/custom_evaluator.py) expose this contract directly.

## Measured outcome

Untouched confirmation, after the recipes and artifacts were locked:

| Vision seed | RegMean worst deficit | After bounded repair | Repaired floors met |
|---|---:|---:|---:|
|11|52.2 points|22.8 points|0/4|
|29|18.3 points|5.3 points|2/4|

The same-start joint baseline meets 2/4 floors on each seed. Repair has a slightly smaller worst deficit, while joint preserves more capabilities on seed 11. There is no unqualified winner. The comparison matches per-candidate gradient examples, steps and teacher calls; total selection/search compute differs. Ablations do not establish benefits of deficit priority, variable rank or protection individually.

Text leet accuracy improves67.2%→78.2% on confirmation, with clean accuracy unchanged at 86.3%. Its earlier selection result71.43% missed the72% floor. Confirmation cannot retroactively approve that artifact.

[Full per-task results and costs](docs/RESULTS.md) · [143 attempted recipes](study/recipe-registry.json) · [Frozen selections](study/deployment-selection.json) · [Protocol](docs/PROTOCOL.md) · [Independent review](research/FINAL_REVIEW.md)

## Evidence and limits

Forty correctness/failure tests cover compatibility, corruption, immutable exports, baseline equations, numeric overflow, protection, rank budgets and rollback. Both deployed static artifacts changed outputs in fresh wheel installations while the OS denied parent-project reads, caches and network: [vision proof](study/independent-vision.json), [text proof](study/independent-text.json). The initial packaging failures remain recorded.

The study trains real specialists from shared small bases. Vision uses three artificial digit domains with a common ten-class output; text uses three rank 4 adapters over fixed character features. Source-disjoint train/calibration/selection/confirmation partitions, two vision seeds, strong spectral/data-informed methods, same-start joint baselines, conflict/insufficient-data/easy controls and four ablations are retained. No external pretrained weights were used. Writer-disjoint digits, template-disjoint SMS, foundation-model transfer and statistical significance are not established.

[Architecture, editable Mermaid](docs/ARCHITECTURE.md) · [Integrations](docs/INTEGRATIONS.md) · [Reproduction](docs/REPRODUCTION.md) · [Demo](docs/media/README.md) · [Launch draft](docs/LAUNCH_DRAFT.md)

Code and locally trained weights are MIT. UCI-derived digit samples remain CC BY 4.0; raw SMS messages are excluded. [Dataset and method attribution](docs/ATTRIBUTION.md) credits Alpaydin/Kaynak, Almeida/Hidalgo, TIES, DARE, TSV, RegMean, mergekit and FusionBench. [Research notes](research/RESEARCH.md) discuss newer overlapping methods; general novelty is not claimed.
