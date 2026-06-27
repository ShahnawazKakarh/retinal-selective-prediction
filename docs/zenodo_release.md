# Zenodo Release Guide

This document walks through publishing a versioned, DOI-backed release on Zenodo and getting it onto ORCID. Mirrors the same workflow used for the SER project.

---

## What you need before starting

- A GitHub account (you already have one)
- A Zenodo account, linked to your GitHub account ([zenodo.org/account/settings/github](https://zenodo.org/account/settings/github))
- A built PDF (`report/retinal-selective-prediction-v1.0.0.pdf` &mdash; see below)
- 15 minutes

---

## Step 1 — Build the PDF locally

```bash
cd ~/retinal-selective-prediction

# Install WeasyPrint if you don't have it
pip install 'weasyprint>=63'

# Build the report
python report/build.py
```

Output: `report/retinal-selective-prediction-v1.0.0.pdf` (cover page + ~12 content pages). Verify it opens cleanly and the dark-theme rendering looks right.

If you want to tweak text before locking the v1.0.0 DOI, edit `report/report.html` and rebuild. Nothing else gets edited.

---

## Step 2 — Enable Zenodo on the GitHub repo

1. Go to [zenodo.org/account/settings/github](https://zenodo.org/account/settings/github)
2. Find `ShahnawazKakarh/retinal-selective-prediction` in the list
3. Toggle the switch to **ON**

This tells Zenodo to listen for GitHub releases on that repo. From now on, every GitHub release creates a new Zenodo version DOI under a single shared Concept DOI.

---

## Step 3 — Tag and release v1.0.0 on GitHub

```bash
cd ~/retinal-selective-prediction

# Tag the current commit
git tag -a v1.0.0 -m "v1.0.0 — first Zenodo release. Single-seed benchmark on APTOS 2019."

# Push the tag
git push origin v1.0.0
```

Then on GitHub:

1. Go to `github.com/ShahnawazKakarh/retinal-selective-prediction/releases`
2. Click **Draft a new release**
3. Select tag `v1.0.0`
4. Title: `v1.0.0 — Single-backbone benchmark on APTOS 2019`
5. Description: paste the abstract from `report/report.html` (or the summary from the README's results table)
6. **Attach the PDF** by drag-dropping `report/retinal-selective-prediction-v1.0.0.pdf` into the assets box
7. Click **Publish release**

Within a minute or two, Zenodo will mint a DOI for this release. Watch your Zenodo dashboard at [zenodo.org/account/settings/github](https://zenodo.org/account/settings/github) &mdash; the new entry appears with a DOI link.

---

## Step 4 — Polish the Zenodo metadata

Zenodo auto-populates metadata from `CITATION.cff` and the GitHub release notes, but a couple of fields are worth editing manually on the Zenodo entry page:

- **Resource type:** Publication &rarr; Technical report
- **Communities:** add `Open Science` and any DR / medical-imaging community you find relevant
- **Keywords:** copy from `CITATION.cff` (already comprehensive)
- **Funding:** leave blank if self-funded, or add a funder name + grant ID if applicable
- **Related identifiers:** the GitHub repo URL is auto-added; you can also add a "supplements" link to your skakarh.com blog post about this work, once written

Click **Save** &mdash; the DOI does not change when you edit metadata.

---

## Step 5 — Update `CITATION.cff` with the real DOI

Once Zenodo gives you the Concept DOI and the v1.0.0 version DOI, edit `CITATION.cff`:

```yaml
# Replace this placeholder
  doi: "10.5281/zenodo.PENDING"
# With the real Concept DOI
  doi: "10.5281/zenodo.XXXXXXX"
```

Use the **Concept DOI** (the parent / "all versions" DOI), not the version-specific one. The Concept DOI is the stable identifier you want on ORCID and in citations &mdash; it resolves to the latest version automatically. The version-specific DOI is for cases where someone wants to cite this exact v1.0.0.

Commit and push:

```bash
git add CITATION.cff
git commit -m "doc: add Zenodo Concept DOI to CITATION.cff"
git push
```

---

## Step 6 — Add the DOI to your ORCID

1. Log into [orcid.org](https://orcid.org)
2. Under **Works**, click **+ Add works** &rarr; **Search &amp; link** &rarr; pick **DataCite** (Zenodo issues DOIs through DataCite)
3. Authorize the connection if asked
4. Your new work should appear in the import list. Click **Add to ORCID record**.

Alternatively, you can add the work manually using the DOI: ORCID &rarr; Add &rarr; Add manually &rarr; paste DOI &rarr; ORCID fetches the metadata.

The same Concept DOI will keep showing up for every future version &mdash; you only do this once.

---

## Step 7 — Verify everything is live

- [ ] DOI resolves: `https://doi.org/10.5281/zenodo.XXXXXXX` opens the Zenodo page
- [ ] PDF downloads from Zenodo
- [ ] GitHub release page shows the PDF attached
- [ ] ORCID profile shows the new work
- [ ] README badge updated (see below)

Add a DOI badge to the README:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

---

## How future versions work

For v1.1.0 (and v2.0.0 later), repeat the same flow:

1. Make the code changes
2. Update `report/report.html` with the new results and `version: 1.1.0` in `CITATION.cff`
3. Rebuild the PDF as `report/retinal-selective-prediction-v1.1.0.pdf`
4. Tag, release, attach PDF

Zenodo automatically files the new version under the **same Concept DOI**. Your ORCID entry stays the same; readers see "this work has multiple versions" and can pick the latest. This is exactly the workflow you used for SER, just applied to a multi-version project.

---

## Release log

| Version | Date | Status | Version DOI | Notes |
|---|---|---|---|---|
| v1.0.0 | 2026-06-13 | ✅ Released | [`10.5281/zenodo.20681524`](https://doi.org/10.5281/zenodo.20681524) | First clean release with the four-method benchmark |
| v1.1.0 | 2026-06-14 | ✅ Released | [`10.5281/zenodo.20695855`](https://doi.org/10.5281/zenodo.20695855) | Added OACSP novel methodology + val/test split fix |
| v1.2.0 | TBD | 🟡 Planned | | Multi-seed variance + IDRiD external validation + OACSP transfer test |
| v2.0.0 | TBD | 🔵 Planned | | Deep Ensembles + Evidential DL + Messidor-2 + equity audit, target IEEE J-BHI |

**Concept DOI** (always resolves to latest): [`10.5281/zenodo.20681415`](https://doi.org/10.5281/zenodo.20681415)

---

## What if I make a mistake on the released metadata?

Zenodo allows editing metadata fields (title, description, keywords) after release without changing the DOI. The PDF file itself, however, is immutable once a DOI is minted &mdash; if you find a typo in the PDF, you'd publish a new minor version (e.g., v1.0.1) rather than mutating v1.0.0.

This is by design: DOIs are supposed to be permanent pointers to specific artifacts.

---

## Common pitfalls

- **Forgetting to attach the PDF to the GitHub release.** Zenodo only captures what's in the release assets at the time of publishing. If you forget, delete the release, re-attach, re-publish.
- **Editing `report.html` after release.** The PDF you released is the artifact &mdash; if you change the source after release, future versions will diverge. Keep the released version frozen in the v1.0.0 tag.
- **Using the version-specific DOI on ORCID.** Use the **Concept DOI** for ORCID and for the CITATION.cff. The Concept DOI auto-resolves to the latest version.
- **License mismatch.** Confirm that `LICENSE` (MIT) and the license dropdown on Zenodo match. They should both be MIT.
