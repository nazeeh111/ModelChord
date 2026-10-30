# v0.2 owner walkthrough

The selected vision artifact passes the frozen required confirmation contract. The text integration runs but fails shifted transfer. Read task rows and acceptance status together; a working application or exported ZIP does not establish general accuracy.

1. Follow the [README installation](../README.md#run-locally), then run `modelchord serve --project . --port 8765` and open [127.0.0.1:8765](http://127.0.0.1:8765). The server uses permitted calibration/selection arrays, not confirmation examples.
2. Keep all three digit specialists selected. Open the compatibility disclosure: common base, 64 normalized pixels, ten ordered digit labels and a common architecture are required. Different representation or ancestry must reject even when shapes match.
3. Select **RegMean**, enable repair and choose **v2 · trained low-rank factors**. Use requested floors 88% clean and 80% left/right/inverted, total rank 8, four stages, 30 steps per trial and a 3% planning buffer. Clean also permits at most 2 points of regression from the fixed ordinary parent. The interface displays requested floors, planning targets and protected effective targets separately.
4. Build. At most six interventions per effective stage are trained, up to 24 trajectories and 720 optimizer updates under these ceilings. The recorded primary run stops after three stages: 18 trajectories, 540 updates and accepted rank 6. Its selection accuracies are clean 92.59%, left 84.44%, right 84.44% and inverted 85.19%. Selection success is a development result; locked confirmation is reported separately in [RESULTS_V02.md](RESULTS_V02.md).
5. Inspect the recipe and report. Check exact source/data identities, effective floors, every candidate, completed accesses, rank and status. Export downloads safe weights, manifest and the build report. A fresh build can have different timing metadata even when its weight hash matches the frozen candidate.
6. In the input panel, load a permitted digit sample, run inference, change pixels and run again. Click or Space cycles a pixel; arrow keys move between pixels. Scores come from the chosen static artifact. Changed scores show responsiveness, not that a modified image still represents the same digit or that every future input meets a floor.
7. Exercise a negative case without changing the study: keep the requested floors and set rank/stages to zero, or request an unattainable target in a new local build. Read the retained frontier and `no acceptable candidate found`. Do not revise the locked recipe after inspecting confirmation.
8. Run the independent applications from the repository root:

```bash
python examples/predict.py experiments/v02/main/vision-seed11-repair/model --input study/demo-vision-inputs.json
python examples/text_app.py experiments/v02/main/text-seed43-repair/model 'Meeting at two.' 'WIN a FREE prize!'
```

These applications load one static artifact, with the full head included. Text preprocessing comes from `examples/text_features.py`. The author-written strings are functionality examples, not evidence that the text model transfers reliably.

Compare the model against joint and fixed-factor controls in the [results](RESULTS_V02.md). Both controls pass the vision11 required contract and are stronger on transformed tasks. The primary repair's extra confirmation planning buffer passes only 3/4 tasks. Seed29's failed planning selection stays rejected despite its later required-floor scores. Text fails all four transferred floors and has no fresh SMS confirmation.

For incompatibilities, run `modelchord check BASE EXPERT...`. Unknown architectures/licenses, altered hashes, changed tokenizer/label order, nonfinite weights and mixed text/vision contracts reject. Never overwrite original source artifacts or the frozen study. New builds live under `exports/`; reproducible research work belongs in a fresh [replay workspace](REPRODUCTION_V02.md).

[Current v0.2 screen captures and short demo](media/v02/README.md) · [API/CLI](API_V02.md) · [Architecture](ARCHITECTURE_V02.md). UI evidence and software isolation checks are separate from empirical contract evidence.
