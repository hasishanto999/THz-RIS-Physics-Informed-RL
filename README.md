# THz-RIS Physics-Informed Reinforcement Learning

MATLAB research code for evaluating reinforcement-learning-based phase control of a reconfigurable intelligent surface (RIS) at **300 GHz**, under near-field propagation, RIS element faults, and imperfect angle observations.

> **Research-code note:** This repository packages the supplied MATLAB scripts, result files, and figures. Before using the numerical results in a paper, rerun the experiments from a clean environment and verify that the reported tables/figures match the saved data and current source code.

## Project overview

The project compares conventional phase-control baselines with PPO-based controllers in a THz RIS link. The shared channel and helper functions are implemented in `thz_lib.m` so that training and evaluation use the same core model.

The supplied scripts describe the following conditions and methods:

- **Carrier frequency:** 300 GHz.
- **RIS:** 16 reflecting elements.
- **Base station array:** 8 elements, using maximum-ratio transmission (MRT).
- **Propagation:** far-field and near-field options; the Stage 9 evaluation uses near-field propagation.
- **Hardware faults:** elements 3 and 12 are configured as stuck at zero phase in the supplied realistic evaluation.
- **Imperfect observations:** angle-estimation noise is included; the robustness script sweeps multiple noise levels.
- **Learning method:** proximal policy optimization (PPO), with different action parameterizations.
- **Baselines:** closed-form Eq. (4), near-field conjugate phase, near-field fault-aware analytic control, and genie/reference bounds where implemented.

## Repository contents

| File | Purpose |
|---|---|
| `thz_lib.m` | Shared channel physics, phase-control helpers, evaluation functions, confidence-interval helper, and self-test. |
| `stage8_multiseed_train.m` | Trains PPO controllers across multiple random seeds and controller/action parameterizations. |
| `stage9_multiseed_eva.m` | Runs the multi-seed comparison experiment and creates comparison outputs. The filename is abbreviated as supplied. |
| `stage9c_angle_error_robustness.m` | Evaluates robustness to angle-estimation error and includes bootstrap-based confidence-interval analysis. |
| `stage8_sigma1_agent_*.mat` | Saved trained-agent artifacts for different agent types and seeds. |
| `stage8_sigma1_stage9_ultimate_results.mat` | Saved Stage 9 comparison results. |
| `stage9c_v2_results.mat` | Saved Stage 9C angle-error robustness results. |
| `stage8_sigma1_training_summary.mat` | Training summary data. |
| `fig_*.png` | Exported figures supplied with the project. |

The `.mat` files are MATLAB data files. Their presence allows inspection of the supplied results without immediately retraining every agent.

## Requirements

- MATLAB.
- **Reinforcement Learning Toolbox** for PPO training and agent operations.
- **Statistics and Machine Learning Toolbox** may be required for statistical routines such as `ttest` and `tinv` used by the robustness analysis.
- Enough disk space and memory for the agent/result `.mat` files.

Exact toolbox requirements can vary by MATLAB release. If MATLAB reports a missing function or toolbox, check the script's call stack and your installed licenses.

## Quick start

1. Download or clone this repository and extract it to a local folder.
2. Open MATLAB and set the current folder to the repository root.
3. Run the shared-library self-test:

   ```matlab
   thz_lib('selftest')
   ```

4. To retrain the agents, open `stage8_multiseed_train.m` and inspect its **USER SETTINGS** before running it. The script comments describe a smoke-test option and the full multi-seed run. Full training can take hours.
5. After the required agent files exist, run the evaluation script:

   ```matlab
   stage9_multiseed_eva
   ```

6. Run the angle-error robustness study:

   ```matlab
   stage9c_angle_error_robustness
   ```

   Review the settings at the top of the script first. It expects the agent files named by its configured file prefix and seed range.

### Important filename/configuration check

The source comments refer to the evaluation script as `stage9_multiseed_eval.m` in some places, but the supplied file is named `stage9_multiseed_eva.m`. Use the actual filename in this repository, or rename it consistently and update any references if you choose to do so.

The Stage 9C script defaults to a specific prefix and seed range. If you use a different set of trained agents, update `FILE_PREFIX`, `TRAIN_SEEDS`, and related settings to match the files that actually exist.

## Reproducibility checklist

Before reporting results:

- Record the MATLAB release and installed toolbox versions.
- Keep the random seeds and evaluation settings fixed when comparing methods.
- Confirm that all methods use the same test angles/noise draws and the same physical channel assumptions.
- Distinguish **variation across trained seeds** from **variation across evaluation trials** when reporting uncertainty.
- Use the bootstrap/seed-aware intervals described in the robustness script where appropriate; do not present per-trial tests as evidence of training-seed generalization.
- Regenerate figures from the saved numeric results rather than relying only on screenshots.
- Verify the final manuscript's units, sample counts, confidence intervals, and statistical claims against the actual `.mat` outputs.

## Model scope and limitations

This is a simulation study, not a hardware measurement campaign. The channel model and assumptions are defined in `thz_lib.m`; inspect that file before interpreting the results. In particular, the comments describe uniform element amplitudes in the near-field model and a compact BS array treated as far-field. These are modeling choices, not universal properties of real THz RIS deployments.

The results depend on the supplied geometry, channel assumptions, fault model, training setup, and evaluation configuration. Do not generalize the reported rates or percentages to other deployments without additional validation.

## Suggested GitHub workflow

1. Create a new GitHub repository.
2. Upload the contents of this folder (not the ZIP file inside another repository).
3. Check that `README.md` appears on the repository home page.
4. Add a license only if you have chosen the terms under which others may reuse your work.
5. If you later publish a paper, add the paper citation and a brief link to the final published version.

## Citation

If you use this code in academic work, cite the associated paper once it is publicly available. Until then, a suitable placeholder is:

> Author(s), “THz-RIS Physics-Informed Reinforcement Learning,” unpublished research code, year.

Replace this placeholder with the correct author list, title, venue, DOI, and repository version when those details are finalized.

## Contact

Add your preferred academic contact or GitHub profile here before making the repository public.
