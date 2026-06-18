# EEG–Video Analysis: C9orf72-Knockout Mice

Longitudinal two-channel EEG (with synchronous video) in *C9orf72*-knockout and wild-type
littermate mice, analysed to characterise a state-resolved trajectory of network dysfunction
across the disease course.

This repository holds the analysis code, derived results, and figures for the manuscript
*"A biphasic trajectory of cortical and hippocampal network dysfunction during REM sleep in
C9orf72-deficient mice"* (in preparation; target: *Brain* / *Brain Communications*).

---

## Study at a glance

- **Model:** *C9orf72*-knockout (KO) vs wild-type (WT) littermates.
- **Recording:** Two-channel EEG with synchronous video.
  - Channel 0 — **CA3** hippocampal *depth* electrode (AP −2.5, ML +3.0, DV −3.0 mm).
  - Channel 1 — **S1/PtA** parietal sensorimotor cortical *surface* electrode (AP −2.0, ML −2.0 mm).
  - Reference — frontal (AP +1.0, ML +1.0 mm).
- **Timepoints (EEG):** 3 m (pre-symptomatic baseline), 4 m (kainic-acid challenge),
  6 m, 7 m, 9 m, 12 m (end-stage).
- **Behaviour:** battery administered at **10 months** (between the 9 m and 12 m EEG sessions):
  novel object recognition, open field, cued/contextual fear conditioning, grip strength.
- **Sampling:** signals down-sampled to 500 Hz, segmented into 4-second epochs, scored as
  wake / NREM / REM from relative band-power criteria.

## Key findings

- **Biphasic beta sign-flip (central result):** relative beta power is elevated in KO at the
  4 m challenge (REM *d* = 1.27, wake *d* = 1.26) and reduced at 12 m (REM *d* = −1.66,
  wake *d* = −1.61); cluster-robust mixed-effects genotype×time interaction *p* = 0.048.
- **Acute REM dysregulation** at challenge: REM theta/delta ratio *d* = 1.97; persistent REM
  theta elevation through 9 m (*d* = 1.76).
- **Sleep-state destabilisation:** progressive NREM→REM transition bias (12 m *d* = 3.85).
- **Genotype-specific PAC null:** no cortical theta–gamma phase-amplitude coupling deficit at
  any timepoint — a divergence from SOD1/FUS models — robust across the full coupling matrix
  and not attributable to a gamma-power confound.
- **Circuit dissociation:** progressively increasing CA3–S1/PtA decorrelation in KO
  (ρ = 0.43, *p* = 0.007) but not WT.
- **Multi-domain behavioural phenotype** at 10 m: reduced grip strength (*d* = −1.23, *p* = 0.045)
  and cued freezing (*d* = −1.84, *p* = 0.012); large-effect trends in recognition memory and
  locomotion.
- **Preserved corticomotor coupling** (animal-level) across the disease course — a supporting
  null indicating the phenotype is specific to REM network organisation rather than a global
  cortex–behaviour breakdown.

---

## Repository layout

```
eeg-video-analysis-c9orf72/
├── src/                  Analysis pipeline (numbered by stage) + shared resolvers
│   ├── abf_paths.py      Canonical path/recording resolver (iter_recordings, video_for, …)
│   ├── 09_behavioral_eeg_integration.py    EEG↔behaviour correlations (3/9/12 m + trajectories)
│   ├── 16_pac_matrix_power_roc.py          PAC matrix, power-confound check, ROC
│   ├── 17_cv_auc_and_staging_validation.py Cross-validated AUC + staging-threshold grid
│   ├── 18_directed_connectivity.py         Transfer entropy + cross-frequency directionality
│   ├── 19_ei_balance_and_criticality.py    Aperiodic (E/I) exponent + criticality
│   ├── 20_hmm_states_and_video_coupling.py HMM state dynamics + corticomotor video coupling
│   └── 20a_pair_videos_by_mtime.py         Pair video files to recordings by modification time
├── data/
│   └── file_inventory_all_timepoints.csv   Source of truth for all recordings (see below)
├── results/              Derived CSVs (group comparisons, correlations, coupling, etc.)
├── figures/              Generated figure PNGs
└── manuscript/           Manuscript (.docx/.md), build script, and Figure 1 generator
```

> Earlier-stage scripts (feature extraction, spectral/coupling/complexity computation,
> sleep-state classification, spindle and coherence analysis) are numbered by pipeline stage
> in `src/`.

---

## Data organisation

The analysis is driven by a single inventory file rather than by folder scanning:

- **`data/file_inventory_all_timepoints.csv`** — one row per recording, with columns:
  `scenario, animal_id, session_id, session, group, timepoint, abf_file, wmv_file,
  abf_path, wmv_path, abf_gb, wmv_gb`.
- **Genotype** is taken from the folder (WT/KO), which is authoritative.
- **Analysis scenarios:**
  - **Scenario A — animal-level** (primary): one value per animal per timepoint.
  - **Scenario B — session-level** (sensitivity): each recording session as the unit.

### Raw data location (not in this repository)

Raw ABF recordings and video are large and live on an **external drive (`F:`)**, not in version
control. `src/abf_paths.py` resolves inventory rows to absolute paths on that drive. The
timepoint folders on `F:` map as:

| Folder on `F:`                              | Timepoint |
|---------------------------------------------|-----------|
| `Baseline (24hrs)_3monthsold_before_KA`     | 3 m       |
| `EEG_After_20KA_i.p._Treatment`             | 4 m       |
| `Baseline (24hrs)_6monthsold_20mgkgKA_Rx`   | 6 m       |
| `Baseline_7month`                           | 7 m       |
| `EEG 10mon`                                 | 9 m       |
| `EEG 1yr`                                   | 12 m      |

---

## Environment & usage

Developed on Windows with a Conda environment named `eeg_video` (Python 3.10).

```bat
conda activate eeg_video
python src\09_behavioral_eeg_integration.py
```

Core dependencies: `numpy`, `scipy`, `pandas`, `statsmodels`, `scikit-learn`, `matplotlib`,
`pyabf` (ABF loading), `fooof`/`specparam` (aperiodic fitting), `opencv-python` (video),
`openpyxl` (behavioural spreadsheets). Most scripts read the inventory, resolve raw files via
`abf_paths.py`, compute features, and write derived CSVs/figures to `results/` and `figures/`.

Key processing constants: `FS_DS = 500` Hz (target sampling rate), `EPOCH_S = 4.0` s,
`MIN_EPOCHS = 20` (minimum epochs per recording-state for inclusion).

---

## Manuscript

The `manuscript/` folder contains the working manuscript (`manuscript_v2_brain.docx` and
`.md`), the document build script, and `make_figure1.py` (which generates the study-design
schematic, Figure 1). Figures 2–6 and the supplementary figures are produced by the analysis
scripts in `src/`.

---

## Reproducibility notes

- Statistics use a pre-specified, FDR-controlled framework with per-timepoint comparisons and
  mixed-effects trajectory modelling (cluster-robust by animal).
- Effect sizes are reported as Cohen's *d* with bootstrap confidence intervals.
- End-stage timepoints (9 m, 12 m) involve small samples; results there are interpreted with
  caution and the within-animal longitudinal design is used to mitigate this.

---

## Authors

Belay [Surname]¹, Tewolde Teklu²\*, Haben Girmay Yhdego³

¹ Tanz Centre for Research in Neurodegenerative Diseases, University of Toronto, Canada
² Axum University, Axum, Ethiopia
³ [Affiliation]

\* Corresponding author.

---

## To complete

- [ ] Confirm total and per-timepoint animal numbers (enrolment/attrition) against the study log.
- [ ] Add original ABF sampling rate and kainic-acid dose to Methods.
- [ ] Choose and add a `LICENSE` (e.g. MIT for code; specify terms for data).
- [ ] Add citation / DOI once the manuscript and any data deposit are public.
- [ ] (Optional) Add a `requirements.txt` or `environment.yml` pinning package versions.
