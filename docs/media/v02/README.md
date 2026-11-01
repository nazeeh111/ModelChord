# v0.2 demonstration

[20-second demonstration](MODELCHORD-v0.2.0.mp4) captions actual completed local workbench captures. It is edited still-state footage, without audio or a continuous-recording claim. The build uses RegMean, v2 repair, rank8,4 stages,30 steps per trial and a3 percentage-point planning buffer. Its exported weights byte-match the primary frozen vision candidate.

The workbench displays selection scores. The separate writer-disjoint confirmation scores are in [v0.2 results](../../RESULTS_V02.md); the left task passes its required80% floor but misses the stronger83% planning target on confirmation. Text transfer fails all four required floors.

`demo-manifest.json` binds capture hashes, captions, output hash, duration and frame count. `render_demo_v02.py` recreates the captioned video. Mobile capture uses390px viewport with no horizontal page overflow.
