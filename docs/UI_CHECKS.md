# Workbench checks

The workbench calls the local catalog, build, sample and prediction endpoints. No benchmark scores or prediction results are hardcoded. It displays per-capability floors alongside ordinary and final quality, retains unmet-floor feedback, and offers artifact export and recorded recipes.

Artifact labels prefer the server's label. Without one, a nonzero accepted correction rank identifies a corrected model; a repair algorithm or tested candidates with zero accepted rank identifies a repair attempt without a correction. An ordinary build is labeled as an ordinary merged model even when it includes an evaluation report.

Static checks:

- `node --check src/modelchord/web/app.js` passes.
- Assets use local routes and system fonts; no external URLs or package dependencies are present.
- API text is inserted using `textContent` or DOM element construction.
- Controls have labels, keyboard focus, and persistent pending/error feedback.
- Pixel editing uses a roving tab stop, arrow navigation, and per-pixel intensity labels.
- Responsive styles include narrow-phone layouts and reduced-motion preferences.

The main agent inspected the current rendered UI and retained `docs/media/workbench-initial.png`. Main-agent browser tests exercise the live build and inference flows separately from these static checks.

The repair budget exposes total correction rank (0–64, default 8) and gradient rounds (0–32, default 4). It sends both numeric values in the build request. The workbench states the fixed 72-candidate limit; unchecking repair disables its budget controls.
