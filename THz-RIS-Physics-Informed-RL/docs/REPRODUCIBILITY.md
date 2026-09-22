# Reproducibility checklist

1. Open the repository root as the MATLAB current folder.
2. Run `src/stage2_channel_model.m`.
3. Run `src/stage7_full_csi_montecarlo.m`.
4. Run `src/stage6_baseline_ris.m` if a fresh Stage 6 result is required.
5. Run `src/stage8_ppo_training.m` for the standard PPO agent, or `src/stage8_multiseed_train.m` for the five-seed experiment.
6. Run the relevant Stage 9 evaluation scripts.

The included `.mat` files are saved outputs from the supplied code package; they are not regenerated during repository packaging. Training is stochastic and can take substantial time.
