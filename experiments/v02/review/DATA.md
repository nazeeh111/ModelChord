# v0.2 confirmation-source preparation

Prepared 2026-10-29. No model was imported, loaded or evaluated. The only computations were parsing, identity checks, class counts and duplicate hashes. Existing v0.1 source, artifacts and ledgers were not changed.

## Vision: ready, 3,823 fresh source images

**Active eligibility ledger:** `../data/vision-collision-v2/vision-confirmation-index.json`. The original raw-only eligibility ledger is preserved as prior evidence. The versioned directory was staged completely and renamed atomically; its manifest binds the prior ledger, prior manifest, raw files, audit script and both new outputs.

Use `../data/raw/optdigits.tra` exclusively for new confirmation. Its 3,823 rows have zero exact-pixel overlaps with **all** 1,797 old source images, and zero internal exact-pixel duplicates. `../data/vision-confirmation-index.json` records every eligible zero-based row, label and image hash. Hash exactly 64 row-major uint8 pixel counts before dividing by sixteen; labels are excluded from duplicate identity. Do not use any of these rows for training, calibration, candidate selection, schedule diagnostics or method changes.

[UCI's official optical-digit page](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) and downloaded `optdigits.names` identify the original TRAIN as 30 writers and TEST as 13 different writers. We reverse the original file roles for this continuation: v0.1 developed on the original TEST; v0.2 confirms on the original TRAIN. This is writer-disjoint **relative to v0.1**, not the original benchmark protocol. Individual writer IDs are unavailable. The official TEST parsed pixels and labels exactly equal current scikit-learn `load_digits`, all 1,797 rows in order. This is stronger than checking only old confirmation rows.

Attribution: Alpaydin, E. and Kaynak, C. (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning Repository, DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49). UCI currently lists CC BY 4.0. Preserve credit, license link and disclose preprocessing.

### Four-view collision audit

Before model metrics, `../data/audit_vision_views.py` generated all four v0.1 views directly on uint8 counts: clean, left shift with zero final column, right shift with zero first column, and inversion `16-x`. It hashed only the 64 resulting pixels. **All 15,292 new view instances were compared against all 7,188 old view instances, across every task pairing. Zero collisions were found.** Whole-source exclusion would remove any new source with any matching old view; none required exclusion, so all 3,823 remain eligible.

There were also **zero transformed-view duplicate groups across distinct new source rows**. All new 15,292 hashes and all old 7,188 hashes are unique. No internal view duplication changes the source denominator. Per-task evaluation would have 3,823 source rows; pooling four task views creates correlated observations and must not be called 15,292 independent source samples. Source-level uncertainty must group four views of each image.

`../data/vision-collision-v2/transformed-view-audit.json` records counts, task-pair matches, excluded IDs, within-new groups and policy. An independent vectorized tuple-of-pixels comparison reproduced the zero-overlap and unique-view counts without hashing. All new bundle hashes verified and prior files remained unchanged. No model loaded or evaluated.

## Text: independent collection, optional transfer only

There is no new balanced confirmation within the old SMS corpus: v0.1's 1,306 balanced examples already include all 653 unique spam messages. A new random partition would reuse observed sources. This conclusion follows from the supplied study provenance, not new reading of raw SMS.

The strongest immediately usable independent **collection** located with explicit redistribution terms is [UCI YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection), also documented by the [authors' laboratory](https://lasid.dcomp.ufscar.br/youtube-spam-collection/). It is **comment spam**, with different authors, settings and spam semantics from SMS. It can test bounded transfer; it cannot establish fresh SMS accuracy or make old SMS confirmation untouched again. Do not silently reuse the SMS floors as claims about this population.

The downloaded five public CSV files contain 1,956 records. Exact text hashing removes 196 repeat occurrences; 1,760 remain (919 ham, 841 spam). No conflicting-label exact groups were found. The class mapping 0=ham, 1=spam matches the official per-video class counts. There are no exact-text SHA256 matches against any of the 1,306 v0.1 balanced SMS source IDs. Comparison does **not** cover unused SMS ham, normalized/near duplicates or shared templates. Raw SMS was not opened. This is independently collected data with no detected exact overlap against used SMS; it is not a proof of semantic independence.

`../data/text-transfer-index.json` identifies eligible records by CSV name and zero-based row; it contains hashes and labels, no comment text or author names. No cohort sampling, partition or model metric has been chosen. Before any inference, freeze whether this is confirmation-only and how imbalance, transformations, floors and pooled/per-video reporting will be handled. Do not tune on any confirmation-only cohort.

Attribution: Alberto, T. and Lochter, J. (2015), YouTube Spam Collection, UCI Machine Learning Repository, DOI [10.24432/C58885](https://doi.org/10.24432/C58885). UCI currently lists CC BY 4.0. Use CONTENT only for input; AUTHOR, DATE and COMMENT_ID are excluded from inference. Keep raw public comments out of promotional examples.

Other candidates were weaker for this immediate protocol: the [Mendeley SMS Phishing corpus](https://data.mendeley.com/datasets/f45bkkt8pr/1) has CC BY 4.0 but explicitly cites the old SMS corpus as a source, so independent provenance is unproven. [ExAIS SMS](https://github.com/AbayomiAlli/SMS-Spam-Dataset) describes a separate community collection but supplies no visible explicit reuse license; it was not downloaded. The [ChatGPT-extended SMS release](https://data.mendeley.com/datasets/kr43v7kdmm/1) mixes the old corpus with synthetic text, so it is unsuitable as independent real-SMS confirmation. None was evaluated.

## Retained identity and reproduction

`../data/source-manifest.json` records source URLs, attribution, licenses, archive identities and every retained raw-file/index hash. Download archives remain disposable under `/private/tmp`; selected original files are retained under `../data/raw/`. To repeat preparation after fetching the same archives, run `.venv/bin/python experiments/v02/data/prepare_sources.py` from the repository root. It never imports MODELCHORD or calls a model.

| Source | Official ZIP SHA256 | Bytes |
| --- | --- | ---: |
| Optical digits | `0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0` | 591292 |
| YouTube spam | `bd6182891adb3cfc8334b82c062176dfbebc563bf0ba07e31c2645f916865a0a` | 163567 |

Raw optical TRAIN SHA256: `e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd`. Original 64-feature integer representation and class order 0–9 are unchanged in the raw file.

Publication boundary: retain all downloaded raw files locally. Publish hashes, eligibility/audit ledgers and preparation scripts; do not publish raw SMS or raw YouTube comments.
