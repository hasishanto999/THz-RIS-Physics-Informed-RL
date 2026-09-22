%% REPOSITORY PATHS
repoRoot = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(repoRoot, 'results');
agentsDir  = fullfile(repoRoot, 'trained_agents');

%% =========================================================
% STAGE 9 HARDWARE: PERFORMANCE COMPARISON WITH DISCRETE PHASES
%
% Runs the standard Stage 9 comparisons plus a novelty test: 2-bit
% hardware quantization, applied to BOTH Stage 7 (closed form) and
% Stage 8 (RL), so we can report which method is more robust to
% realistic finite-resolution RIS hardware - not just how Stage 8
% alone degrades.
% =========================================================
clear; clc; close all;

%% =========================================================
% SECTION A: RICIAN COMPARISON (Stage 6 vs Stage 7)
% =========================================================
fprintf('\n============================================\n');
fprintf(' STAGE 9A: RICIAN COMPARISON (Stage 6 vs Stage 7)\n');
fprintf(' K=1, 100 trials, same channel draws both stages\n');
fprintf('============================================\n\n');

load(fullfile(resultsDir, 'stage6_results.mat'), 'Rate_stage6_all');
load(fullfile(resultsDir, 'stage7_montecarlo_results.mat'), 'Rate_optimized_all');
Rate_stage7_all = Rate_optimized_all;

fprintf('Stage 6 (theta=0 baseline) Rate = %.4f +/- %.4f Gbps\n', mean(Rate_stage6_all), std(Rate_stage6_all));
fprintf('Stage 7 (joint BS+RIS opt) Rate = %.4f +/- %.4f Gbps\n', mean(Rate_stage7_all), std(Rate_stage7_all));

improvement_A = Rate_stage7_all - Rate_stage6_all;
fprintf('Improvement (paired)            = %.4f +/- %.4f Gbps (%.1f%% mean gain)\n', ...
    mean(improvement_A), std(improvement_A), 100*mean(improvement_A)/mean(Rate_stage6_all));
fprintf('Improvement positive in %d/%d trials\n', sum(improvement_A > 0), numel(improvement_A));

%% =========================================================
% SECTION B: LOS-ONLY COMPARISON WITH HARDWARE CONSTRAINTS
% =========================================================
fprintf('\n============================================\n');
fprintf(' STAGE 9B: LOS-ONLY COMPARISON (Includes 2-bit Hardware)\n');
fprintf('============================================\n\n');

load(fullfile(agentsDir, 'stage8_trained_agent.mat'), 'agent', 'THzEnv');

actor_obj = getActor(agent);

num_trials_B = 100;
rng(2024);
theta_RU_samples = 10 + rand(num_trials_B,1) * 70;

Rate_s6_los = zeros(num_trials_B,1);
Rate_s7_los = zeros(num_trials_B,1);
Rate_s8_los = zeros(num_trials_B,1);
Rate_s8_discrete = zeros(num_trials_B,1);
Rate_s7_discrete = zeros(num_trials_B,1);   % NEW: Stage 7 under same hardware constraint

n_BS  = (0:THzEnv.N_BS-1).';
a_BS  = exp(1i*2*pi*THzEnv.d_BS/THzEnv.lambda * n_BS*sind(THzEnv.theta_BS));

n_RIS = (0:THzEnv.N_RIS-1).';
a_BR  = exp(1i*2*pi*THzEnv.d_RIS/THzEnv.lambda * n_RIS*sind(THzEnv.theta_BR));

PL_BR_dB = 20*log10(4*pi*THzEnv.d_BR/THzEnv.lambda) + 4.343*THzEnv.k_abs*THzEnv.d_BR;
PL_RU_dB = 20*log10(4*pi*THzEnv.d_RU/THzEnv.lambda) + 4.343*THzEnv.k_abs*THzEnv.d_RU;
G_BR = 10^(-PL_BR_dB/10);
G_RU = 10^(-PL_RU_dB/10);

num_bits = 2;
M = 2^num_bits;
step_size = 2*pi / M;

for i = 1:num_trials_B
    theta_RU = theta_RU_samples(i);
    a_RU = exp(1i*2*pi*THzEnv.d_RIS/THzEnv.lambda * n_RIS*sind(theta_RU));
    H_BR = sqrt(G_BR) * (a_BR * a_BS');
    h_RU = sqrt(G_RU) * a_RU;

    % ---- Stage 6: theta = 0, BS MRT-matched to a_BS ----
    Theta6 = eye(THzEnv.N_RIS);
    w6 = a_BS/sqrt(THzEnv.N_BS);
    h6 = h_RU' * Theta6 * H_BR * w6;
    Rate_s6_los(i) = THzEnv.B * log2(1 + THzEnv.Pt_W*abs(h6)^2/THzEnv.Noise_W) / 1e9;

    % ---- Stage 7: closed-form joint optimum ----
    c_genie = conj(h_RU) .* a_BR;
    theta7 = -angle(c_genie);
    Theta7 = diag(exp(1i*theta7));
    h_BS7 = h_RU' * Theta7 * H_BR;
    w7 = h_BS7' / norm(h_BS7');
    h7 = h_BS7 * w7;
    Rate_s7_los(i) = THzEnv.B * log2(1 + THzEnv.Pt_W*abs(h7)^2/THzEnv.Noise_W) / 1e9;

    % ---- Stage 8: trained PPO agent (Continuous Phase) ----
    obs = {[sind(theta_RU); cosd(theta_RU)]};
    out = evaluate(actor_obj, obs);
    slope8 = out{1} * pi;
    theta8 = n_RIS * slope8;
    Theta8 = diag(exp(1i*theta8));
    h_BS8 = h_RU' * Theta8 * H_BR;
    w8 = h_BS8' / norm(h_BS8');
    h8 = h_BS8 * w8;
    Rate_s8_los(i) = THzEnv.B * log2(1 + THzEnv.Pt_W*abs(h8)^2/THzEnv.Noise_W) / 1e9;

    % ---- NOVELTY: Stage 8 with 2-bit Discrete Phase ----
    theta_discrete = round(theta8 / step_size) * step_size;
    Theta_discrete = diag(exp(1i*theta_discrete));
    h_BS_discrete = h_RU' * Theta_discrete * H_BR;
    w_discrete = h_BS_discrete' / norm(h_BS_discrete');
    h_discrete = h_BS_discrete * w_discrete;
    Rate_s8_discrete(i) = THzEnv.B * log2(1 + THzEnv.Pt_W*abs(h_discrete)^2/THzEnv.Noise_W) / 1e9;

    % ---- Same 2-bit quantization applied to Stage 7's closed-form phases ----
    % Needed for a fair claim: "RL is/isn't more robust to hardware
    % quantization than the closed form" requires quantizing BOTH, not
    % just Stage 8. Without this, there's no baseline to compare against.
    theta7_discrete = round(theta7 / step_size) * step_size;
    Theta7_discrete = diag(exp(1i*theta7_discrete));
    h_BS7_discrete = h_RU' * Theta7_discrete * H_BR;
    w7_discrete = h_BS7_discrete' / norm(h_BS7_discrete');
    h7_discrete = h_BS7_discrete * w7_discrete;
    Rate_s7_discrete(i) = THzEnv.B * log2(1 + THzEnv.Pt_W*abs(h7_discrete)^2/THzEnv.Noise_W) / 1e9;
end

fprintf('Stage 6 (theta=0)         Rate = %.4f +/- %.4f Gbps\n', mean(Rate_s6_los), std(Rate_s6_los));
fprintf('Stage 7 (closed form)     Rate = %.4f +/- %.4f Gbps\n', mean(Rate_s7_los), std(Rate_s7_los));
fprintf('Stage 8 (Continuous AI)   Rate = %.4f +/- %.4f Gbps\n', mean(Rate_s8_los), std(Rate_s8_los));
fprintf('Stage 7 (2-bit hardware)  Rate = %.4f +/- %.4f Gbps\n', mean(Rate_s7_discrete), std(Rate_s7_discrete));
fprintf('Stage 8 (2-bit hardware)  Rate = %.4f +/- %.4f Gbps\n', mean(Rate_s8_discrete), std(Rate_s8_discrete));

pct_continuous = 100 * mean(Rate_s8_los) / mean(Rate_s7_los);
pct_discrete_vs_own_continuous_s8 = 100 * mean(Rate_s8_discrete) / mean(Rate_s8_los);
pct_discrete_vs_own_continuous_s7 = 100 * mean(Rate_s7_discrete) / mean(Rate_s7_los);
pct_discrete_vs_genie = 100 * mean(Rate_s8_discrete) / mean(Rate_s7_los);

fprintf('\nStage 8 AI (Continuous) achieves %.1f%% of the Genie-Optimal (Stage 7) rate.\n', pct_continuous);
fprintf('Stage 7 retains %.1f%% of its own continuous rate under 2-bit quantization.\n', pct_discrete_vs_own_continuous_s7);
fprintf('Stage 8 retains %.1f%% of its own continuous rate under 2-bit quantization.\n', pct_discrete_vs_own_continuous_s8);
fprintf('Stage 8 (2-bit) achieves %.1f%% of the fully continuous genie-optimal rate.\n', pct_discrete_vs_genie);

if pct_discrete_vs_own_continuous_s8 > pct_discrete_vs_own_continuous_s7
    fprintf('\n=> RL-learned phases are MORE robust to 2-bit quantization than the closed form.\n');
else
    fprintf('\n=> RL-learned phases are NOT more robust to 2-bit quantization than the closed form.\n');
end

%% =========================================================
% SAVE + PLOTS
% =========================================================

save(fullfile(resultsDir, 'stage9_hardware_results.mat'), ...
    'Rate_stage6_all', 'Rate_stage7_all', 'improvement_A', ...
    'Rate_s6_los', 'Rate_s7_los', 'Rate_s8_los', ...
    'Rate_s7_discrete', 'Rate_s8_discrete', 'theta_RU_samples');

figure('Color', 'w', 'Position', [100, 100, 1000, 450]);

subplot(1,2,1);
boxplot([Rate_stage6_all, Rate_stage7_all], 'Labels', {'Stage 6 (Baseline)', 'Stage 7 (Optimum)'});
ylabel('Data Rate (Gbps)');
title('Section A: Rician (K=1)');
grid on;

subplot(1,2,2);
boxplot([Rate_s6_los, Rate_s7_los, Rate_s7_discrete, Rate_s8_los, Rate_s8_discrete], ...
    'Labels', {'Stage 6', 'Stage 7 (cont.)', 'Stage 7 (2-bit)', 'Stage 8 (cont.)', 'Stage 8 (2-bit)'});
ylabel('Data Rate (Gbps)');
title('Section B: LOS-only, continuous vs. 2-bit hardware');
grid on;

fprintf('\n============================================\n');
fprintf(' STAGE 9 (WITH HARDWARE NOVELTY) COMPLETE\n');
fprintf('============================================\n');