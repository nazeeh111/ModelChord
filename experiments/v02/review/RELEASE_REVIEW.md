# Independent v0.2 release-readiness review

Reviewed 2026-10-29 against the original attached objective, current working-tree source, README, all five V02 documents, frozen lock/results, prior scientific review, product tests, capture evidence and release preparation. No source, frozen file, registration, model or cohort was changed. No network, publication, commit, training or confirmation inference was performed. Only this new review file was written. Local synthetic checks explicitly used `PYTHONPATH=src`, because the project's existing `.venv` still contains an older installed package.

## Decision

No unresolved consequential source or scientific-claim defect was found in the checked version. One CLI input-validation finding was corrected during this review and independently checked below. The scoped source is ready for the owner's final packaging/publication work. **This review does not mark the public release, remote CI, clean-clone installation or final anonymous-download inference proof as passed.** Those are still separately executing delivery gates, not implied by this decision.

The original objective is satisfied only within the openly declared small-classifier scope. Vision11 has a supported bounded retention success; text transfer is a failed empirical contract despite its runnable software integration. Reused expert seeds/tasks, artificial digit domains, unequal FLOPs, tied ablations, repeated selection and historical controls remain explicit limitations. No universal guarantee, repair superiority, causal mechanism benefit, foundation-model composition or memory advantage is supported.

## Finding and verified correction

**Resolved P2: v2 CLI silently cast invalid NPZ labels before validation.** The initially reviewed loader used `.long()` for y and `.float()` for x in the common CLI data path, allowing fractional labels to become different integer targets. This could silently corrupt the user's declared calibration/evaluation task while the strict v2 API appeared to validate it.

The checked correction at `src/modelchord/cli.py:52-57` now preserves NPZ tensor dtypes for `repair-v2`, retaining conversion only for the historical `repair` command. The unchanged strict API check at `src/modelchord/repair_v2.py:78-81` rejects non-int64 labels and non-float32 inputs before export. New regression cases are at `tests/test_v02_product.py:117` (fractional float32/float64 labels and float64 x). Independently executed:

```text
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_v02_product.py -k 'preserves_invalid_data_types or no_search_budget_keeps_requested_contract'
6 passed, 39 deselected in 0.64s
```

A separate synthetic fractional-label command against current source also returned exit 2 with `Invalid bounded finite CPU classification batch`. No cohort examples were involved. Parent-reported full-suite evidence is 104 passed after the correction; this reviewer independently ran the six targeted checks above rather than representing that full run as its own.

## Evidence inspected and recomputed

- Independently rehashed all **272 frozen paths**, with zero mismatches. `src/modelchord/repair_v2.py` remains byte-identical to frozen `experiments/v02/repair_search.py`. The CLI correction does not alter numerical method, selected weights or frozen study evidence.
- Recomputed every recorded participant's integer-count accuracy, required deficit vector, fraction of floors met, worst deficit and acceptance conjunction from the stored lock/results. All 35 participant rows agree. This was arithmetic over existing results, not new inference.
- Vision11 repair remains 3483/3823 clean, 3155/3823 left, 3192/3823 right and 3250/3823 inverted. Required floors and regression bounds pass; the stronger left planning buffer misses confirmation. Seed29's later required-floor success does not override its failed planning selection. Text repair remains rejected at 1042/1760, 1064/1760, 1126/1760 and 1023/1760 for clean/uppercase/leet/underscores.
- Read the factor search and reference code, including frozen preceding weights/biases, zero-effect factor initialization, six interventions per stage, fixed original protected reference, lexicographic deficit objective, rollback, static folding and access counters (`repair_v2.py:42-160,163-238`). No hidden full-weight optimization was found in factor search. Joint training is explicitly separated.
- Current CLI stores requested floors, applied buffer, effective protected selection targets and required-floor fraction in `build-report.json` (`cli.py:69-72`). The workbench downloadable ZIP separately includes those values (`server.py:75-95`), and UI computes required versus selection checks separately (`web/app.js:71-100`). Zero effective stages disable the buffer in both routes.
- Read product tests for negative requests, budget caps, no-search behavior and downloaded report accounting. Source preserves loopback binding, Host/Origin checks, catalog paths, safe NPZ loading and typed failure responses. These checks are software behavior evidence, not adversarial ML robustness.
- Inspected the current overview capture visually. It labels selection data, distinguishes requested/planning targets and displays the actual selected build's development scores. The media README explicitly calls the demo edited still-state footage. The workbench evidence records changed pixels and a 390px no-page-overflow check; those UI interactions were recorded by the builder, not independently re-executed here.
- Read reproduction/build/runtime evidence and scripts. Exact 12-weight replay is development reproduction, not fresh confirmation. Runtime scopes distinguish loaded forward from safe open/validation/inference and exclude process/import/HTTP startup. Dense heads count in storage. No efficiency claim is justified by the noisy observations.
- Read `scripts/verify_release_v02.py`: it is designed to download all eight anonymous assets, compare SHA256/bytes, reinstall the downloaded wheel and execute both static artifacts with source, clone, caches and network denied. Source inspection alone is not a passed execution. The existing `publication/v02/preliminary-runtime-verification.json` explicitly describes an older copied-dependency wheel and is not adequate final proof for corrected CLI bytes.

## Delivery checks still owned by the parent

Before calling the requested public release complete, record successful remote CI for the exact published commit; clean-clone installation and quickstart; downloaded asset hashes/bytes and links; final current-wheel and both artifact changed-input runs under parent/cache/network denial; repository metadata and preservation of the existing v0.1 history/release. No new scientific acceptance condition or extra approval gate is introduced here. Text's failed transfer status and all scope limits must remain visible in release notes, assets and walkthrough.

## Version binding

Working-tree review base commit: `36adfda70544d7af2ff9ffe22f70cd4a8fc48fa8` (v0.1 history retained; v0.2 changes were uncommitted at review). SHA256 identities below bind the actual checked bytes. Subsequent edits or rebuilt publication assets require their own final identity checks; this is not a review of unseen future bytes.

| Checked path | SHA256 |
| --- | --- |
| `README.md` | `3eb8b9ed9b118542f5767ac90034ab3f97914cd87fad3736f8f17874f08703ee` |
| `docs/API_V02.md` | `325752567b9d9df120382db4215e8fcdaed88ac1e659a19018afacd42bcd1a9f` |
| `docs/ARCHITECTURE_V02.md` | `f8a9f80ae2257d86b10c83ff8a207f1283bbe5cd2d64e2ae48e31c8767721ba0` |
| `docs/REPRODUCTION_V02.md` | `0f78b54d8aa06548d3b7f29731dc8b70705d2d4c07401f0a7e7748bed6beebf6` |
| `docs/RESULTS_V02.md` | `4a12d9057a9214b154b491b6b29bbd45af465009da11d9bdd2334032cf80cdde` |
| `docs/WALKTHROUGH_V02.md` | `81857b006c1107a1d7dcbf6a614c6ec39fc1d63f16fd4125ccadca2753e8a71d` |
| `src/modelchord/cli.py` | `a3e8e94454517b9c454cc7a0790f3379874cb94f488095d348ae7b14b178173f` |
| `src/modelchord/server.py` | `ae465167e6febfd5f4f5455a1b77fa1c81784df8fe6d5e44846603299106dfed` |
| `src/modelchord/repair_v2.py` | `baeafcdd1b618920181e8c2aad77e68681dced7f7f16561fc813c75aa3e24a88` |
| `src/modelchord/web/app.js` | `9f95da482c14d08770e8e75c6097d12caceb4d8a8c463c00e4106297eecf4981` |
| `src/modelchord/web/index.html` | `9fe50e1cd7ecd523224363ada655ae627ac4a9d39cdaf4e7be196f0bbb882a92` |
| `src/modelchord/web/style.css` | `dceef47fb3068a511756436f1fb2504e7920ba670e97339bde0cbd1732b03262` |
| `tests/test_v02_product.py` | `b6d02f228adb2d7ccaa078f2e75b57b1b8212da2036b5e5dded70e4f88a2a9c2` |
| `pyproject.toml` | `efaacc3f5f0fb1a5c3d6a7b88b5aaf0b941003e65d3005bf6c2a1a2d71f89de9` |
| `.github/workflows/ci.yml` | `5f2686d2bf6ac086b0596f76d5e1c32ed2deb679b49d3ea0a8d91fe26d7192de` |
| `scripts/ci_smoke.py` | `17158264b3af8a279d42adca756a3f90b5bb02637002eb0897e8f13bcea7f584` |
| `scripts/reproduce_v2.py` | `1224e7b17a5b374c9fe1c39af90ca1643a43fde7107ad3fd677e67cf598e07c1` |
| `scripts/prepare_v2_development.py` | `d3177be172c36b99a17c251d6d7e8063337c9356b96173405c3a2691b22b8512` |
| `scripts/package_release_v02.py` | `72ed94afeba1164aef2a98308241b0c0347f6a4668e7ee7a41dedb7ef808ce0e` |
| `scripts/verify_release_v02.py` | `58f7e7c92523a3309316b28ab933037dc5a596efbcb1a8d7653bbd69b61eb605` |
| `experiments/v02/confirmation-lock.json` | `8a776503289ebf8ce2df1beb69a8bd5e27f8e437beaff488d1f7eec648d00f80` |
| `experiments/v02/confirmation-results.json` | `d2ebf138dff6113b493b3b8683abace37a710488713eae1620ba41c1dc8f77dd` |
| `experiments/v02/review/FINAL_REVIEW.md` | `2f44d42780d1ab8c108881941e3bbb5dcabe51c1e0391563a288f14f3b34b720` |
| `experiments/v02/reproduction-check.json` | `15c35252dc5da36cde7e6aff684db7c42535c7e8b64bf21f63597352c14b13e1` |
| `experiments/v02/build-profile.json` | `c4535a88bbb3ae50019e97cab4214bd1530786dcf2f70548979ac5bd6262c20f` |
| `experiments/v02/runtime-profile.json` | `ff21e3a2454ddbb990e543181d62f6efa57b6291d9bf0f6a589e90ddb6d40a23` |
| `docs/media/v02/overview.png` | `45f1ac4c6c8ee86487a8b98922418b5cb2911aee181f195a69d7ad80ee9dea87` |
| `docs/media/v02/demo-manifest.json` | `c0ddacd76236f204d03773fdad7e72f2482a6e084e798294fb1d5fe7f5b95cae` |
