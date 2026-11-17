# CIS 5300 Milestone 1 working package

Prepared November 1, 2026 from the approved three-author proposal.

The [current course page](https://www.cis.upenn.edu/~myatskar/teaching/cis5300_fa26/term-project.html) lists **November 4, 2026** for M1 and requires a scheduled mentor meeting, a report of at most two pages covering three papers and collected data, `data.md`, and a gzipped data archive. The page gives no M1 time of day.

## Files to review and submit

`submission/` contains:

- `milestone1.pdf`: two-page report with three paper reviews and the data description.
- `milestone1.tex`: editable source for the same report.
- `data.md`: source links, exact counts, schema, real example, split and evaluation details.
- `data.tar.gz`: the complete prepared data and manifests; no sample-only substitution.
- `data/`: the unpacked copy for local use.

The source proposal has not been changed. No files have been uploaded or submitted, and no mentor has been contacted.

## Remaining team actions

1. Review the three summaries and confirm they reflect the team's reading and intended approach.
2. Confirm/schedule the mentor meeting. No meeting details were supplied, so the report makes no claim that one is scheduled. Discuss the scope of the two extensions and the M2 evaluation wording with the mentor: the proposal reserves the project test set for final evaluation, while the generic M2 instructions request baseline test scores. Do not tune on test scores either way.
3. Submit the PDF, `data.md`, and `data.tar.gz` through the course's normal process. The `.tex` is the editable companion.

## Reproducibility

The scripts use Python's standard library. From this directory:

```sh
python3 scripts/fetch_sources.py
python3 scripts/prepare_data.py
python3 scripts/package_data.py
```

The first command uses the network and records current source hashes. For exact reconstruction of this prepared release, retain the downloaded `source_data/` files and recorded `references/source_manifest.json`; the preparation script checks those hashes before processing. Do not refresh sources and silently call a different release identical.

To compile the report with a TeX installation:

```sh
tectonic --outdir submission submission/milestone1.tex
```

The actual build was compiled and visually reviewed. `references/` contains the three papers, source text, course-page snapshot, upstream documentation/evaluator, and fetch manifest. `source_data/` preserves the four unmodified data downloads. `submission/data/validation.json` records checks over the full prepared corpus.

No model training, test inference, automatic submission, or custom augmentation was performed for this milestone. M2 implements baselines/scoring; M3 generates custom augmentation.
