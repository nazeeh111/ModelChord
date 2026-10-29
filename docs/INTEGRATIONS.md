# Two static integrations

## Vision application

`examples/predict.py` consumes one exported artifact and JSON `[batch,64]` inputs. Features are row-major8×8 optical digit pixels, divided by16 from the original0–16 values. Ordered classes are digit strings0–9. The suite's shifted/inverted domains have identical feature/output contracts; the runtime does not take a task or expert ID.

```bash
python examples/predict.py artifacts/vision-seed11-v2/repair-strongest --input study/demo-vision-inputs.json
```

The released example inputs come from UCI digits under CC BY 4.0. They are for running the frozen model after confirmation, not new tuning data. The model is13,098 parameters, including its full classifier. This is a controlled digit-domain demonstration, not general image classification.

## Same-base text adapters

`examples/text_features.py` supplies a versioned, deterministic256-dimensional signed UTF-8 byte unigram/bigram representation. It uses BLAKE2b, no learned vocabulary, and L2 normalization. It is case-sensitive. `examples/text_app.py` passes those features into one static MLP with ordered classes`ham,spam`.

```bash
python examples/text_app.py artifacts/text/repaired 'Meeting at two.' 'WIN a FREE prize!'
```

The specialist integrations train rank 4 additive factors over frozen shared base weights, on uppercase, leetspeak and underscore domains. Safe adapter factors/metadata and dense-folded experts are retained. The folding audit verifies exact logits against the adapted experts on three changed vector inputs each. The final repair is folded into dense safe-tensor weights; the application loads no parent base, specialist adapter, teacher or ensemble. The full model, all adapter artifacts and all heads count toward reported storage. This is SMS classification, not LLM composition or a universal spam detector.

The local apps are separate examples; text and vision models are never combined. Both deployment choices remain unaccepted under the frozen selection gate. Dataset/feature/license identities and exact data source/split hashes are bound in the study lock and artifacts. See ATTRIBUTION.md for the official datasets and research credit.

`examples/custom_evaluator.py` demonstrates replacing raw accuracy with balanced class accuracy over present classes. The floor meaning must change explicitly with the evaluator. User-provided evaluators are ordinary trusted Python callables, not remotely loaded recipes.
