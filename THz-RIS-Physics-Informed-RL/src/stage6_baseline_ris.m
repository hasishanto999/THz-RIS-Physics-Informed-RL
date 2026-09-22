clear;
clc;

%% REPOSITORY PATHS
repoRoot = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(repoRoot, 'results');
agentsDir  = fullfile(repoRoot, 'trained_agents');


%% =========================================================
% STAGE 6: RIS PHASE = 0 BASELINE (canonical, theta=0)
%
% This is NOT a separate simulation - it reuses the theta=0 / no-RIS-
% shaping baseline that was already computed inside stage7_montecarlo.m,
% under the same parameters (0.15 m distances, N_BS=8) and the same 100
% channel realizations as the Stage 7 result. Recomputing it separately
% with different distances/antenna counts (as an earlier draft did, at
% 10 m) would make it a different system than Stage 7/8, and any
% "Stage 7 improves on Stage 6 by X dB" claim in Stage 9 would not be a
% real comparison.
%
% Run stage7_montecarlo.m first if stage7_montecarlo_results.mat doesn't
% exist yet.
% =========================================================

load(fullfile(resultsDir, 'stage7_montecarlo_results.mat'), ...
    'Pr_reference_dBm_all', 'SNR_reference_dB_all', 'Rate_reference_all', 'num_trials');

fprintf('\n============================================\n');
fprintf(' STAGE 6: RIS PHASE = 0 BASELINE (theta=0)\n');
fprintf(' (%d trials, same channel draws as Stage 7)\n', num_trials);
fprintf('============================================\n\n');

fprintf('Received Power = %.2f +/- %.2f dBm\n', mean(Pr_reference_dBm_all), std(Pr_reference_dBm_all));
fprintf('SNR            = %.2f +/- %.2f dB\n',  mean(SNR_reference_dB_all), std(SNR_reference_dB_all));
fprintf('Data Rate      = %.4f +/- %.4f Gbps\n', mean(Rate_reference_all), std(Rate_reference_all));

fprintf('\n============================================\n');
fprintf(' STAGE 6 COMPLETE\n');
fprintf('============================================\n');

%% Save under its own filename too, for clarity when building Stage 9

Pr_stage6_dBm_all  = Pr_reference_dBm_all;
SNR_stage6_dB_all  = SNR_reference_dB_all;
Rate_stage6_all    = Rate_reference_all;

save(fullfile(resultsDir, 'stage6_results.mat'), 'Pr_stage6_dBm_all', 'SNR_stage6_dB_all', 'Rate_stage6_all', 'num_trials');
fprintf('Saved to stage6_results.mat\n');