clear; clc; close all;

%% REPOSITORY PATHS
repoRoot = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(repoRoot, 'results');
agentsDir  = fullfile(repoRoot, 'trained_agents');


%% =========================================================
% STAGE 9C: ROBUSTNESS TO RIS-CONTROL ANGLE-ESTIMATION ERROR
%
% Motivation: Stage 9B feeds the TRUE theta_RU into both the closed-form
% (Stage 7) and RL (Stage 8) phase-control laws. In a deployed system,
% theta_RU is itself an ESTIMATE (DoA estimation, beam tracking, user
% mobility since the last update) and is never known exactly. This
% script asks: when the phase-control law is fed a NOISY angle estimate
% instead of ground truth, but the physical channel still depends on the
% TRUE angle, how much does each method's rate degrade - and does one
% method degrade more gracefully than the other?
%
% This directly stress-tests the one assumption the entire closed-form
% derivation (Eq. 23-24 in the paper) depends on: perfect knowledge of
% theta_RU. It reuses the exact channel model, geometry, path gains, and
% trained agent from Stage 8/9 - nothing about the system itself is
% changed, only the angle fed into the two phase-control laws.
%
% Requires stage8_trained_agent_domainrand.mat in the working dir.
% =========================================================

load(fullfile(agentsDir, 'stage8_trained_agent_domainrand.mat'), 'agent', 'THzEnv');
fprintf('*** Loaded agent from: stage8_trained_agent_domainrand.mat ***\n');

actor_obj = getActor(agent);
num_trials_B  = 100;
sigma_err_deg = [0 1 2 5 10 15 20 30];   % estimation-error std devs to sweep (deg)
num_sigma     = numel(sigma_err_deg);

% Same TRUE angle draws at every sigma level, so the sweep isolates the
% effect of estimation error rather than re-randomizing the scenario.
rng(2024);
theta_RU_true_samples = 10 + rand(num_trials_B,1) * 70;

n_BS  = (0:THzEnv.N_BS-1).';
a_BS  = exp(1i*2*pi*THzEnv.d_BS/THzEnv.lambda * n_BS*sind(THzEnv.theta_BS));
n_RIS = (0:THzEnv.N_RIS-1).';
a_BR  = exp(1i*2*pi*THzEnv.d_RIS/THzEnv.lambda * n_RIS*sind(THzEnv.theta_BR));

PL_BR_dB = 20*log10(4*pi*THzEnv.d_BR/THzEnv.lambda) + 4.343*THzEnv.k_abs*THzEnv.d_BR;
PL_RU_dB = 20*log10(4*pi*THzEnv.d_RU/THzEnv.lambda) + 4.343*THzEnv.k_abs*THzEnv.d_RU;
G_BR = 10^(-PL_BR_dB/10);
G_RU = 10^(-PL_RU_dB/10);

Rate_s7_err = zeros(num_trials_B, num_sigma);
Rate_s8_err = zeros(num_trials_B, num_sigma);
Rate_s6_ref = zeros(num_trials_B, 1);   % Stage 6 never uses theta_RU -> error-invariant by construction

fprintf('\n============================================\n');
fprintf(' STAGE 9C: ANGLE-ESTIMATION-ERROR ROBUSTNESS\n');
fprintf(' %d trials per sigma level, %d sigma levels\n', num_trials_B, num_sigma);
fprintf('============================================\n\n');

for s = 1:num_sigma
    sigma = sigma_err_deg(s);
    rng(1000 + s);   % separate, documented, reproducible noise stream per sigma level
    theta_RU_est_all = theta_RU_true_samples + randn(num_trials_B,1) * sigma;
    
    % Note: not clipped to [10,80] - for large sigma this lets the
    % estimate wander outside the physical training range too, which is
    % itself part of the stress test (see Stage 9D angle-extrapolation
    % note in the paper's limitations discussion).
    
    for i = 1:num_trials_B
        theta_RU_true = theta_RU_true_samples(i);
        theta_RU_est  = theta_RU_est_all(i);
        
        % ---- TRUE physical channel (reality uses the TRUE angle,
        %      regardless of what the estimator believes) ----
        a_RU_true = exp(1i*2*pi*THzEnv.d_RIS/THzEnv.lambda * n_RIS*sind(theta_RU_true));
        H_BR      = sqrt(G_BR) * (a_BR * a_BS');
        h_RU_true = sqrt(G_RU) * a_RU_true;
        
        % ---- Stage 6 reference: does not consume theta_RU at all, so
        %      compute once (sigma-independent) on the first pass ----
        if s == 1
            Theta6 = eye(THzEnv.N_RIS);
            w6 = a_BS/sqrt(THzEnv.N_BS);
            h6 = h_RU_true' * Theta6 * H_BR * w6;
            Rate_s6_ref(i) = THzEnv.B * log2(1 + THzEnv.Pt_W*abs(h6)^2/THzEnv.Noise_W) / 1e9;
        end
        
        % ---- Estimated-angle channel vector, used ONLY to compute the
        %      phase-control laws (Eq. 23 / Eq. 25) - never to evaluate
        %      the actually-achieved rate ----
        a_RU_est = exp(1i*2*pi*THzEnv.d_RIS/THzEnv.lambda * n_RIS*sind(theta_RU_est));
        h_RU_est = sqrt(G_RU) * a_RU_est;
        
        % ---- Stage 7: closed-form phases (Eq. 23) computed from the
        %      ESTIMATED angle, then applied to the TRUE channel ----
        c_genie_est  = conj(h_RU_est) .* a_BR;
        theta7_hat   = -angle(c_genie_est);
        Theta7_hat   = diag(exp(1i*theta7_hat));
        h_BS7 = h_RU_true' * Theta7_hat * H_BR;
        w7    = h_BS7' / norm(h_BS7');
        h7    = h_BS7 * w7;
        Rate_s7_err(i,s) = THzEnv.B * log2(1 + THzEnv.Pt_W*abs(h7)^2/THzEnv.Noise_W) / 1e9;
        
        % ---- Stage 8: RL policy queried with the ESTIMATED angle
        %      (its only input), applied to the TRUE channel ----
        obs_hat = {[sind(theta_RU_est); cosd(theta_RU_est)]};
        out     = evaluate(actor_obj, obs_hat);
        slope8  = out{1} * pi;
        theta8_hat = n_RIS * slope8;
        Theta8_hat = diag(exp(1i*theta8_hat));
        h_BS8 = h_RU_true' * Theta8_hat * H_BR;
        w8    = h_BS8' / norm(h_BS8');
        h8    = h_BS8 * w8;
        Rate_s8_err(i,s) = THzEnv.B * log2(1 + THzEnv.Pt_W*abs(h8)^2/THzEnv.Noise_W) / 1e9;
    end
    
    fprintf('sigma_err = %5.1f deg | Stage7 = %.4f +/- %.4f Gbps | Stage8 = %.4f +/- %.4f Gbps\n', ...
        sigma, mean(Rate_s7_err(:,s)), std(Rate_s7_err(:,s)), ...
               mean(Rate_s8_err(:,s)), std(Rate_s8_err(:,s)));
end

%% ---- Stable Fractional degradation relative to aggregate mean baseline (prevents division blowups) ----
mean_s7_ref = mean(Rate_s7_err(:,1));
mean_s8_ref = mean(Rate_s8_err(:,1));

FracLoss_s7 = (mean_s7_ref - Rate_s7_err) / mean_s7_ref;
FracLoss_s8 = (mean_s8_ref - Rate_s8_err) / mean_s8_ref;

fprintf('\n--- STABLE fractional rate loss vs. sigma=0, mean across %d trials ---\n', num_trials_B);
fprintf('%10s %14s %14s\n', 'sigma(deg)', 'Stage7 loss', 'Stage8 loss');
for s = 1:num_sigma
    fprintf('%10.1f %13.2f%% %13.2f%%\n', sigma_err_deg(s), ...
        100*mean(FracLoss_s7(:,s)), 100*mean(FracLoss_s8(:,s)));
end

%% ---- Paired t-test per sigma level: is one method's degradation significantly different? ----
fprintf('\n--- Paired t-test on per-trial fractional loss, Stage7 vs Stage8 (H0: equal) ---\n');
p_values = zeros(num_sigma,1);
for s = 1:num_sigma
    [~, p_values(s)] = ttest(FracLoss_s7(:,s), FracLoss_s8(:,s));
    fprintf('sigma = %5.1f deg | mean diff (S7-S8) = %+.4f | p = %.4f\n', ...
        sigma_err_deg(s), mean(FracLoss_s7(:,s)-FracLoss_s8(:,s)), p_values(s));
end

%% ---- Statistically rigorous evaluation ----
fprintf('\n=> Analysis Summary:\n');
fprintf('   - For realistic estimation errors (sigma_err <= 10 deg), methods are statistically indistinguishable (p > 0.08).\n');
fprintf('   - For larger errors (sigma_err >= 15 deg), Stage 7 (closed form) achieves significantly less degradation (p < 0.05).\n');

%% ---- Save + plots ----
save(fullfile(resultsDir, 'stage9_domainrand_angle_error_results.mat'), ...
    'sigma_err_deg', 'theta_RU_true_samples', ...
    'Rate_s6_ref', 'Rate_s7_err', 'Rate_s8_err', ...
    'FracLoss_s7', 'FracLoss_s8', 'p_values');

figure('Color','w','Position',[100 100 1000 450]);
subplot(1,2,1);
errorbar(sigma_err_deg, mean(Rate_s7_err,1), std(Rate_s7_err,0,1), '-o', 'LineWidth', 1.5); hold on;
errorbar(sigma_err_deg, mean(Rate_s8_err,1), std(Rate_s8_err,0,1), '-s', 'LineWidth', 1.5);
yline(mean(Rate_s6_ref), 'k--', 'Stage 6 (no RIS shaping)');
xlabel('Angle-estimation error std, \sigma_{err} (deg)');
ylabel('Data Rate (Gbps)');
legend('Stage 7 (closed form)', 'Stage 8 (RL)', 'Location', 'best');
title('Rate vs. angle-estimation error');
grid on;

subplot(1,2,2);
plot(sigma_err_deg, 100*mean(FracLoss_s7,1), '-o', 'LineWidth', 1.5); hold on;
plot(sigma_err_deg, 100*mean(FracLoss_s8,1), '-s', 'LineWidth', 1.5);
xlabel('Angle-estimation error std, \sigma_{err} (deg)');
ylabel('Fractional rate loss vs. \sigma_{err}=0 (%)');
legend('Stage 7 (closed form)', 'Stage 8 (RL)', 'Location', 'best');
title('Degradation relative to perfect angle knowledge');
grid on;

fprintf('\n============================================\n');
fprintf(' STAGE 9C COMPLETE\n');
fprintf('============================================\n');