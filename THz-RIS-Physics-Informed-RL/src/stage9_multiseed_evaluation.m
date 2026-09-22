clear; clc; close all;

%% REPOSITORY PATHS
repoRoot = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(repoRoot, 'results');
agentsDir  = fullfile(repoRoot, 'trained_agents');


%% =========================================================
% STAGE 9E: MULTI-SEED RL EVALUATION vs. CLOSED FORM
%
% Loads every stage8_trained_agent_seed<N>.mat produced by
% stage8_multiseed_train.m and evaluates each on the SAME 100 true
% theta_RU draws (identical rng(2024) seed to Stage 9B/9C), against the
% same closed-form (Stage 7) baseline. Reports:
%   - per-seed mean rate and % of genie-optimal (variance across the
%     100 evaluation trials, as in Stage 9B)
%   - variance ACROSS THE 5 TRAINING SEEDS themselves - i.e., how much
%     the headline "RL achieves X% of genie-optimal" number would have
%     changed if training had happened to converge to a different local
%     solution.
%
% Requires stage8_trained_agent_seed42.mat ... seed46.mat (or whichever
% TRAIN_SEEDS were used) in the working directory.
% =========================================================

TRAIN_SEEDS = [42 43 44 45 46];
num_seeds   = numel(TRAIN_SEEDS);
num_trials_B = 100;

% ---- Load geometry from the first seed's THzEnv (identical across seeds
%      by construction - only the trained weights differ) ----
tmp = load(fullfile(agentsDir, sprintf('stage8_trained_agent_seed%d.mat', TRAIN_SEEDS(1))), 'THzEnv');
THzEnv = tmp.THzEnv;

rng(2024);   % SAME evaluation draws as Stage 9B/9C, for comparability
theta_RU_samples = 10 + rand(num_trials_B,1) * 70;

n_BS  = (0:THzEnv.N_BS-1).';
a_BS  = exp(1i*2*pi*THzEnv.d_BS/THzEnv.lambda * n_BS*sind(THzEnv.theta_BS));

n_RIS = (0:THzEnv.N_RIS-1).';
a_BR  = exp(1i*2*pi*THzEnv.d_RIS/THzEnv.lambda * n_RIS*sind(THzEnv.theta_BR));

PL_BR_dB = 20*log10(4*pi*THzEnv.d_BR/THzEnv.lambda) + 4.343*THzEnv.k_abs*THzEnv.d_BR;
PL_RU_dB = 20*log10(4*pi*THzEnv.d_RU/THzEnv.lambda) + 4.343*THzEnv.k_abs*THzEnv.d_RU;
G_BR = 10^(-PL_BR_dB/10);
G_RU = 10^(-PL_RU_dB/10);

% ---- Stage 7 closed-form baseline: computed ONCE (agent-independent) ----
Rate_s7_los = zeros(num_trials_B,1);
for i = 1:num_trials_B
    theta_RU = theta_RU_samples(i);
    a_RU = exp(1i*2*pi*THzEnv.d_RIS/THzEnv.lambda * n_RIS*sind(theta_RU));
    H_BR = sqrt(G_BR) * (a_BR * a_BS');
    h_RU = sqrt(G_RU) * a_RU;

    c_genie = conj(h_RU) .* a_BR;
    theta7  = -angle(c_genie);
    Theta7  = diag(exp(1i*theta7));
    h_BS7   = h_RU' * Theta7 * H_BR;
    w7      = h_BS7' / norm(h_BS7');
    h7      = h_BS7 * w7;
    Rate_s7_los(i) = THzEnv.B * log2(1 + THzEnv.Pt_W*abs(h7)^2/THzEnv.Noise_W) / 1e9;
end

% ---- Stage 8: evaluate every trained agent on the same trials ----
Rate_s8_by_seed = zeros(num_trials_B, num_seeds);

fprintf('\n============================================\n');
fprintf(' STAGE 9E: MULTI-SEED RL EVALUATION\n');
fprintf('============================================\n\n');

for k = 1:num_seeds
    seed = TRAIN_SEEDS(k);
    fname = fullfile(agentsDir, sprintf('stage8_trained_agent_seed%d.mat', seed));
    tmp = load(fname, 'agent');
    actor_obj = getActor(tmp.agent);

    for i = 1:num_trials_B
        theta_RU = theta_RU_samples(i);
        a_RU = exp(1i*2*pi*THzEnv.d_RIS/THzEnv.lambda * n_RIS*sind(theta_RU));
        H_BR = sqrt(G_BR) * (a_BR * a_BS');
        h_RU = sqrt(G_RU) * a_RU;

        obs = {[sind(theta_RU); cosd(theta_RU)]};
        out = evaluate(actor_obj, obs);
        slope8 = out{1} * pi;
        theta8 = n_RIS * slope8;
        Theta8 = diag(exp(1i*theta8));
        h_BS8 = h_RU' * Theta8 * H_BR;
        w8 = h_BS8' / norm(h_BS8');
        h8 = h_BS8 * w8;
        Rate_s8_by_seed(i,k) = THzEnv.B * log2(1 + THzEnv.Pt_W*abs(h8)^2/THzEnv.Noise_W) / 1e9;
    end

    pct_of_genie_this_seed = 100 * mean(Rate_s8_by_seed(:,k)) / mean(Rate_s7_los);
    fprintf('seed %3d | mean rate = %.4f +/- %.4f Gbps (across %d eval trials) | %.1f%% of genie-optimal\n', ...
        seed, mean(Rate_s8_by_seed(:,k)), std(Rate_s8_by_seed(:,k)), num_trials_B, pct_of_genie_this_seed);
end

%% ---- Variance ACROSS TRAINING SEEDS ----

seed_mean_rates = mean(Rate_s8_by_seed, 1);              % 1 x num_seeds
seed_pct_genie  = 100 * seed_mean_rates / mean(Rate_s7_los);

fprintf('\n--- Variance across %d independent training runs ---\n', num_seeds);
fprintf('Stage 7 (closed form, agent-independent) rate = %.4f +/- %.4f Gbps\n', ...
    mean(Rate_s7_los), std(Rate_s7_los));
fprintf('Stage 8 (RL) mean rate across seeds            = %.4f +/- %.4f Gbps\n', ...
    mean(seed_mean_rates), std(seed_mean_rates));
fprintf('Stage 8 %% of genie-optimal across seeds         = %.2f +/- %.2f %%\n', ...
    mean(seed_pct_genie), std(seed_pct_genie));
fprintf('Range across seeds: [%.2f%%, %.2f%%] of genie-optimal\n', ...
    min(seed_pct_genie), max(seed_pct_genie));

%% ---- Save + plot ----
save(fullfile(resultsDir, 'stage9_multiseed_results.mat'), ...
    'TRAIN_SEEDS', 'theta_RU_samples', 'Rate_s7_los', 'Rate_s8_by_seed', ...
    'seed_mean_rates', 'seed_pct_genie');

figure('Color','w','Position',[100 100 700 420]);
bar(TRAIN_SEEDS, seed_pct_genie); hold on;
yline(100, 'k--', 'Genie-optimal (Stage 7)');
xlabel('Training seed');
ylabel('% of genie-optimal rate');
title(sprintf('RL performance across %d independent training runs', num_seeds));
grid on;

fprintf('\n============================================\n');
fprintf(' STAGE 9E COMPLETE\n');
fprintf('============================================\n');
