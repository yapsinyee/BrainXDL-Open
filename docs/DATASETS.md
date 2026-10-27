# Datasets

BrainXDL-Open does not distribute research data, prepared graph folds,
participant identifiers, or checkpoints.

The BSD 3-Clause software licence applies to project-owned source code only. It
does not override dataset licences, consent conditions, governance approvals,
or access agreements.

## PsiConnect

The workflows expect locally authorized fMRI time-series and cleaned EEG files.
Access must be obtained through the study's approved data-governance process.

Expected naming is BIDS-like and includes subject and session entities. The MM
loader pairs modalities by subject/session/task. Session-to-label mapping in
the existing implementation is:

- `ses-01` → positive class / psilocybin
- `ses-02` → control class

All sessions belonging to one participant must remain in the same split.

## ABIDE

The ST scripts include a downloader for public ABIDE PCP derivatives and AAL
ROI time series. Dataset terms and preprocessing provenance must be reviewed by
the user before a result is presented as thesis reproduction.

The downloader is a convenience entry point, not redistribution of ABIDE data.
Users remain responsible for the dataset's current access and usage terms.

## COBRE

The package provides reusable COBRE static/dynamic loading support. Raw scans,
study downloads, trained checkpoints, and campaign-specific outputs are not
distributed.

## Local layout

Paths are supplied through command-line arguments. A typical ignored layout is:

```text
data/
  psiconnect/
    fmri_timeseries/
    eeg_clean/
  st_folds/
```

Never commit material under `data/`.
