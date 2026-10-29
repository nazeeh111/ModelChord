# Owner walkthrough

MODELCHORD is an experimental builder with measured negative results. The downloadable vision and text artifacts do not have an accepted frozen selection contract. Use them to inspect the mechanism and frontier, not as certified all-capability classifiers.

1. Follow the README install commands and run `modelchord serve --project .`. The local workbench has three compatible digit specialists from the same base. Inspect the contract disclosure to see the input normalization, common architecture and ordered digit labels.
2. Choose task floors. The study used88% clean,80% shifted/inverted, with clean protected against more than2 points of regression relative to the ordinary merge. Set rank 8/rounds 4. These controls bound correction size and gradient rounds; at most72 candidate interventions are tested.
3. Select RegMean. It consumes permitted calibration activations. Enable repair and build. Read every task row and the worst shortfall. The example improves inverted accuracy but returns `no acceptable candidate found`. It does not silently turn a partial repair into an accepted build.
4. Inspect the recipe. It records source identities, method, exact calibration/selection hashes, effective protected floors, candidate attempts and accepted corrections. Export downloads a ZIP containing `manifest.json`, `model.safetensors` and the build report. The model weights include the complete classification head.
5. In the input panel, load a permitted sample, run inference, change a pixel with Space or click, and run again. Arrow keys move between pixels. Scores come from the selected artifact. A changed score is inference responsiveness, not proof the changed image retained its semantic digit label.
6. Run `examples/predict.py` against the selected exported directory. Run `examples/text_app.py` with the text artifact and author-written strings. Neither application requires parent experts, a router, network or teacher.
7. Read RESULTS.md before comparing methods. The same-start joint baseline has a different tradeoff; the ablations are mixed or unhelpful. The easy-control floors are deliberately custom and weak. Text's later confirmation success cannot override selection failure.

For compatibility failures, use `modelchord check BASE EXPERT...` before a build. Different text and vision families reject even if someone changes shapes to look similar. Unknown licenses/architectures, altered hashes, changed tokenizer/labels and nonfinite tensors also reject. Source files are never overwritten; choose a new export directory for each build.

The source study is frozen. Reproduce in a fresh folder instead of changing any locked artifact or reusing the observed confirmation set for tuning. Empirical floors apply to the declared finite sample, not all future behavior.
