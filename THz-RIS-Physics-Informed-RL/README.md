# Physics-Informed Reinforcement Learning for Joint BS-RIS Beamforming at 300 GHz

MATLAB simulation code, trained PPO agents, and saved results for the research study:

> **A Physics-Informed Reinforcement Learning Approach to Joint Base-Station and RIS Beamforming for 300 GHz Terahertz Links: Closed-Form Optimization, Learned Control, and Hardware-Quantization Robustness**

## Overview

This repository implements a MATLAB-based 300 GHz THz communication framework comparing a closed-form joint base-station (BS) and reconfigurable intelligent surface (RIS) beamforming solution with a physics-informed PPO reinforcement-learning controller. The workflow also includes multi-seed training, angle-estimation-error robustness, domain-randomized training, and finite-resolution RIS hardware quantization.

### Main system settings

- Carrier frequency: **300 GHz**
- BS antennas: **8**
- RIS elements: **16**
- Bandwidth: **10 GHz**
- LOS THz propagation model
- Free-space path loss + molecular absorption
- Molecular absorption based on the ITU-R P.676-12 line-by-line calculation implemented in Stage 2
- PPO-based RIS control
- 2-bit hardware phase-quantization evaluation

## Repository structure

```text
THz-RIS-Physics-Informed-RL/
├── README.md
├── .gitignore
├── src/
│   ├── stage2_channel_model.m
│   ├── stage6_baseline_ris.m
│   ├── stage7_full_csi_montecarlo.m
│   ├── stage8_ppo_training.m
│   ├── stage8_multiseed_train.m
│   ├── stage8_domain_randomized_train.m
│   ├── stage8_training_diagnostic.m
│   ├── stage9_hardware_quantization.m
│   ├── stage9_angle_error_robustness.m
│   ├── stage9_angle_error_outofrange_diagnostic.m
│   ├── stage9_multiseed_evaluation.m
│   └── stage9_domainrand_angle_error_robustness.m
├── results/
│   ├── README.md
│   ├── stage6_results.mat
│   ├── stage9_angle_error_results.mat
│   ├── stage9_hardware_results.mat
│   └── stage9_multiseed_results.mat
├── trained_agents/
│   ├── README.md
│   ├── stage8_trained_agent.mat
│   ├── stage8_trained_agent_domainrand.mat
│   └── stage8_trained_agent_seed42.mat ... seed46.mat
├── figures/
└── docs/
```

## Recommended execution order

### 1. Channel / absorption calculation

Run:

```matlab
src/stage2_channel_model.m
```

This generates `results/stage2_results.mat`.

### 2. Closed-form Monte Carlo benchmark

Run:

```matlab
src/stage7_full_csi_montecarlo.m
```

This generates `results/stage7_montecarlo_results.mat`.

### 3. Stage 6 baseline summary

Run:

```matlab
src/stage6_baseline_ris.m
```

It reads the Stage 7 trial-level file and saves `results/stage6_results.mat`.

### 4. PPO training

For the standard single agent:

```matlab
src/stage8_ppo_training.m
```

For the five independent seeds:

```matlab
src/stage8_multiseed_train.m
```

For observation-noise domain randomization:

```matlab
src/stage8_domain_randomized_train.m
```

Training can be computationally expensive because the supplied training configuration uses 1500 maximum episodes.

### 5. Diagnostics and robustness

Standard angle-error robustness:

```matlab
src/stage9_angle_error_robustness.m
```

Out-of-range diagnostic:

```matlab
src/stage9_angle_error_outofrange_diagnostic.m
```

Multi-seed evaluation:

```matlab
src/stage9_multiseed_evaluation.m
```

Hardware quantization:

```matlab
src/stage9_hardware_quantization.m
```

Domain-randomized robustness evaluation:

```matlab
src/stage9_domainrand_angle_error_robustness.m
```

## Reproducibility notes

The scripts use repository-relative paths for saved results and trained agents. Personal Windows paths from the original development environment have been removed.

The multi-seed training workflow uses seeds **42, 43, 44, 45, and 46**. The evaluation scripts use the documented random seeds in the original code to keep trial draws comparable across experiments.

## Included data and generated data

The supplied ZIP contained 23 original files. The repository preserves the supplied MATLAB result files and trained agents while reorganizing them into `results/` and `trained_agents/`.

`stage7_montecarlo_results.mat` and `stage2_results.mat` are generated outputs rather than files supplied in the ZIP, so they are not fabricated in this repository. Run the corresponding Stage 2 and Stage 7 scripts to regenerate them.

## MATLAB requirements

The code uses MATLAB Reinforcement Learning Toolbox functionality including PPO agents and training APIs. Additional standard MATLAB functionality is used for numerical simulation and plotting. Use a MATLAB release compatible with the Reinforcement Learning Toolbox APIs used by the scripts.

## Citation

If this code is used in academic work, cite the associated paper. Replace this section with the final bibliographic entry and DOI once the publication record is available.

## Author

**Haseebul Haq**

American International University-Bangladesh (AIUB)
