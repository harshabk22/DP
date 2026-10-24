# AGENTS.md

## What this repo is
- Research and deliverables for a B.Tech design project (course codes E1CSA313 / E1ITA306, "Design Project - 1", mentor Dr. M. Selvam). No code, no build/test/lint, and no version control (there is no `.git`) — copy a file before destructive rewrites.
- The implementation is **not here**. It is a separate project ("MessMate": FastAPI `backend/` with pytest + alembic, Flutter `mobile/`, an `eval/` harness, and `MESSMATE_SPEC.md`). Last seen at `~/Documents/DP-App_Prototype` — do **not** assume it is next door to this repo; locate it by searching for `MESSMATE_SPEC.md` or `DP-App_Prototype`. Requests that need code changes belong there; don't add manifests, scripts, or CI to this repo.

## Artifacts and which one is current
- `literature_review.md` — the source of truth. §3 maps the 20 papers to `papers/` paths, §7 is the recommended architecture, §10 is the reference list.
- `smart_meal_analyzer_paper_analysis_table.docx` — one row per paper `[1]`–`[20]`, columns: Paper title / Key contributions & methodology / Advantages / Disadvantages & limitations.
- `papers/NN_YYYY_Author_ShortTitle.pdf` — the 20 verified full texts (~51 MB total; extract or grep, don't read whole files).
- Deck lineage (26 slides, **newest is current**): `clear-table-review-original-headings` → `clear-table-review-updated` → `clear-table-review-updated  -  Repaired` → `ppt dp` → `ppt dp - with visuals` → `ppt-REV-2.pptx` (formerly `ppt dp - with visuals - Paper Titles.pptx`). Only edit the newest; the rest are superseded drafts. Keep the layout/theme of `Presentation - Template - E1CSA313 - E1ITA306 - Design Project -1 - Dr. M.Selvam.pptx`.

## Paper numbers 1–20 are a shared cross-artifact key
The same number appears in the filename prefix, the §3 table row, and the docx `[n]` row. Add, remove, or renumber a paper in all three places or every cross-reference breaks.

## Source-verification rules (review §2 — project requirements, not preferences)
- Final peer-reviewed version only. No arXiv, no author manuscripts, no preprint-only or predatory-venue records.
- The DOI must resolve and match OpenAlex metadata; no retraction, withdrawal, or expression of concern.
- Years 2022–2026 inclusive; "2026" means published on or before the review date, 23 Sep 2026.
- A workshop/preprint version plus a later journal version counts as one paper.
- Paper 17 (`17_2025_Zhang_MOPI_HFRS_NSF_Accepted_Manuscript.pdf`) is the documented exception: the NSF Public Access Repository copy of the ACM SIGKDD paper. Keep the file and the §2.5 note that explains it.
- Citation style: keep the numbered IEEE-like form of §10 (initials, title, venue, vol/issue/pages, year, DOI link) and the docx provenance prefixes ("Final publisher version of record", "Official CVF accepted version; final IEEE record", "Official NSF PAR copy; ACM KDD publication").

## Claim discipline
- No project results, no clinical or health-outcome claims. Slide 21 labels its chart "Literature-reported error signals — not project results"; keep the per-study literature values (Zhao 15.68%, Moyen 27.4%, Papathanail 27.4%, Shonkoff max 38.3%, Vedovelli 45.6% sMAPE) separate from anything this team measured.
- Slide 26 tracks the submission gate: final similarity check pending, similarity report to be attached before submission.

## Ignore
- `.playwright-mcp/` — scratch output from the 23 Sep search session (page snapshots, console logs, PDF downloads). It holds an author manuscript (`16-2025-Yin-FoodLMM-Author-Manuscript.pdf`) that is not part of the verified 20, so never cite from here.
