# Build contract and API

The v0.1 architecture registry supports bounded float32 ReLU MLP classifiers, common-base dense specialists, and low-rank adapters folded into those classifiers. Unknown architectures, tied tensors, differing label order, tokenizer, preprocessing, ancestry, precision and source-license policies are rejected. This is deliberately narrower than accepting arbitrary same-shaped checkpoints. An expert's declared base identity is an attestation from its producer, not cryptographic proof of training history. The supplied common base's actual tensor digest is verified.

`Artifact.open(path)` validates a content-bound manifest and safe tensor weights. `artifact.predict(float32_batch)` loads only that static artifact and rejects empty, malformed, nonfinite or numerically overflowing input/output. Class lists are explicit in `artifact.manifest['architecture']['classes']`.

```python
from modelchord import Artifact, merge, repair, Capability, BuildBudget
base = Artifact.open('artifacts/vision-seed11-v2/base')
experts = [Artifact.open(f'artifacts/vision-seed11-v2/expert-{t}')
           for t in ['left', 'right', 'inverted']]
ordinary = merge(base, experts, 'exports/ordinary', method='task_arithmetic', scale=.5)
# calibration and selection map names to distinct (float32_x, long_y) batches.
final, report = repair(ordinary, calibration, selection,
    [Capability('clean', .88, protected=True, max_regression=.02),
     Capability('left', .80), Capability('right', .80), Capability('inverted', .80)],
    'exports/repaired', budget=BuildBudget(rank=8, rounds=4))
print(report['status'])
```

Use fresh output directories. Export never overwrites an existing artifact. Source bytes are revalidated during streamed reads. Candidate changes are applied to copies; rejected steps leave the preceding model intact. Ordinary merges load expert tensors one layer at a time. Repairs fit in this registry's bounded memory but do not offer streamed backpropagation for giant models.

The repair objective is lexicographic: worst nonnegative accuracy deficit, sum of deficits, then mean cross entropy. Effective floors for protected tasks also include ordinary-model accuracy minus the permitted regression. Gradients on permitted calibration batches propose rank-limited matrix directions. Actual interventions on distinct selection data determine acceptance. Only the chosen low-rank correction is folded into weights; no full-model optimizer is used by repair. Candidate search and repeated selection consume data and compute and may overfit the selection partition. The untouched confirmation set provides a separate final check. Rank accounting is an upper bound on the sum of accepted per-layer update ranks, not the rank of the final dense model.

`modelchord.evaluate.Evaluator` is the evaluator extension protocol. Supply `evaluator=` to `repair` with callable `(model,x,y) -> {'accuracy': float, 'cross_entropy': float, ...}`. `accuracy` must be a higher-is-better score in [0,1] and the returned loss must be finite. The built-in evaluator reports accuracy, cross entropy, ten-bin expected calibration error and example count. For a different capability metric, explicitly define a compatible bounded score rather than changing the meaning of the floor silently. Original source ID and semantic duplicate separation is the caller's responsibility; API batch hashes catch identical reused batches and bind exact bytes to the report.

CLI:

```bash
modelchord inspect ARTIFACT
modelchord check BASE EXPERT...
modelchord merge BASE EXPERT... --output NEW_PATH --method task_arithmetic --scale .5
modelchord repair ORDINARY --data study/workbench.npz --capabilities capabilities.json --output NEW_PATH
modelchord predict ARTIFACT input.json
modelchord serve --project . --port 8765
```

The repair NPZ uses `cal_x_NAME/cal_y_NAME` and `x_NAME/y_NAME` for distinct calibration/selection batches. Capability JSON is an array of `name`, `floor`, optional `protected`, optional `max_regression`. RegMean is available through the typed API and workbench because it needs per-expert permitted calibration arrays. Static inference has no router, teacher, ensemble, downloads or parent-model access. The CLI and API never load arbitrary remote code or pickle.

The loopback server exposes only its configured catalog and generated exports; it rejects foreign Origin/Host headers, path traversal and arbitrary artifact paths. It is a local development workbench without authentication or public hosting. It must not be publicly exposed.
