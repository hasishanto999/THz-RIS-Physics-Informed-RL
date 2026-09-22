clear;
clc;
close all;

%% REPOSITORY PATHS
repoRoot = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(repoRoot, 'results');
agentsDir  = fullfile(repoRoot, 'trained_agents');


%% =========================================================
% STAGE 7 (MONTE CARLO): JOINT BS + RIS OPTIMIZATION
%
% Change vs. previous version: averaged over multiple independent Rician
% channel realizations (num_trials) instead of a single fixed rng(1)
% draw. A single-seed result can't support a "Method X beats Method Y"
% claim in Stage 9 - this reports mean +/- std across draws instead.
%
% k_abs is now the ITU-R P.676-12 line-by-line value computed in Stage 2
% (stage2_channel_model.m) for fc=300 GHz, standard atmosphere: k_abs =
% 1.198e-3 Np/m (vs. the earlier 0.005 Np/m placeholder). At d=0.15 m
% this changes total path loss by <0.005 dB per link - well below the
% Monte Carlo noise floor - so no numerical result here changes
% materially; this is a correctness/consistency fix, not a re-tune.
% =========================================================

%% 1. SYSTEM PARAMETERS (unchanged from before)

fc = 300e9;
c = 3e8;
lambda = c/fc;

B = 10e9;
Pt_dBm = 20;
Pt_W = 10^((Pt_dBm-30)/10);

N_BS = 8;
N_RIS = 16;

d_BS = lambda/2;
d_RIS = lambda/2;

D_RIS = (N_RIS-1)*d_RIS;
d_FF  = 2*D_RIS^2/lambda;

d_BR = 0.15;
d_RU = 0.15;

if d_BR < d_FF || d_RU < d_FF
    warning(['Link distance below Fraunhofer distance (%.4f m) - ' ...
             'far-field steering-vector model is not valid here.'], d_FF);
end

T = 290;
k_B = 1.380649e-23;
Noise_W = k_B*T*B;
Noise_dBm = 10*log10(Noise_W)+30;

k_abs = 1.198e-3;   % ITU-R P.676-12 line-by-line value from Stage 2 (fc=300 GHz,
                     % standard atmosphere) - see header note

PL_BR = 20*log10(4*pi*d_BR/lambda) + 4.343*k_abs*d_BR;
PL_RU = 20*log10(4*pi*d_RU/lambda) + 4.343*k_abs*d_RU;

G_BR = 10^(-PL_BR/10);
G_RU = 10^(-PL_RU/10);

theta_BS = 20;
theta_BR = 20;
theta_RU = 20;

n_BS = (0:N_BS-1).';
a_BS = exp(1i*2*pi*d_BS/lambda * n_BS*sin(theta_BS*pi/180));

n_RIS = (0:N_RIS-1).';
a_BR = exp(1i*2*pi*d_RIS/lambda * n_RIS*sin(theta_BR*pi/180));
a_RU = exp(1i*2*pi*d_RIS/lambda * n_RIS*sin(theta_RU*pi/180));

K_factor = 1;

%% 2. MONTE CARLO CONFIGURATION

num_trials   = 100;   % independent channel realizations
num_restarts = 50;    % kept per-trial, same as before
inner_iters  = 30;

Pr_reference_dBm_all = zeros(num_trials,1);
Pr_optimized_dBm_all = zeros(num_trials,1);
SNR_reference_dB_all = zeros(num_trials,1);
SNR_optimized_dB_all = zeros(num_trials,1);
Rate_reference_all    = zeros(num_trials,1);
Rate_optimized_all    = zeros(num_trials,1);

theta_reference = zeros(N_RIS,1);
Theta_reference = diag(exp(1i*theta_reference));
w_reference = a_BS/sqrt(N_BS);

fprintf('\n============================================\n');
fprintf(' STAGE 7 (MONTE CARLO): JOINT BS + RIS OPTIMIZATION\n');
fprintf('============================================\n');
fprintf('Running %d independent channel realizations...\n', num_trials);

for trial = 1:num_trials

    rng(trial);   % different, but reproducible, seed per trial

    H_BR_NLOS = (randn(N_RIS,N_BS) + 1i*randn(N_RIS,N_BS))/sqrt(2);
    h_RU_NLOS = (randn(N_RIS,1)    + 1i*randn(N_RIS,1))/sqrt(2);

    H_BR = sqrt(G_BR) * ( ...
           sqrt(K_factor/(K_factor+1)) * (a_BR*a_BS') + ...
           sqrt(1/(K_factor+1))        * H_BR_NLOS );

    h_RU = sqrt(G_RU) * ( ...
           sqrt(K_factor/(K_factor+1)) * a_RU + ...
           sqrt(1/(K_factor+1))        * h_RU_NLOS );

    %% Stage 6 reference (this trial's channel)

    h_reference = h_RU' * Theta_reference * H_BR * w_reference;
    reference_gain = abs(h_reference)^2;

    %% Joint optimization (alternating block-coordinate ascent)

    best_gain  = reference_gain;
    best_w     = w_reference;
    best_theta = theta_reference;

    for r = 1:num_restarts

        if r == 1
            theta = theta_reference;
        else
            theta = 2*pi*rand(N_RIS,1);
        end

        Theta = diag(exp(1i*theta));
        h_BS = h_RU' * Theta * H_BR;
        w = h_BS';
        norm_w = norm(w);
        if norm_w > 0
            w = w/norm_w;
        else
            w = w_reference;
        end

        for it = 1:inner_iters
            c_iter = conj(h_RU) .* (H_BR*w);
            theta = -angle(c_iter);
            Theta = diag(exp(1i*theta));

            h_BS = h_RU' * Theta * H_BR;
            w = h_BS';
            norm_w = norm(w);
            if norm_w > 0
                w = w/norm_w;
            else
                w = w_reference;
            end
        end

        h = h_RU' * Theta * H_BR * w;
        gain = abs(h)^2;

        if gain > best_gain
            best_gain = gain;
            best_w = w;
            best_theta = theta;
        end
    end

    Theta_best = diag(exp(1i*best_theta));
    h_optimized = h_RU' * Theta_best * H_BR * best_w;
    optimized_gain = abs(h_optimized)^2;

    %% Metrics for this trial

    Pr_reference_W = Pt_W*reference_gain;
    Pr_optimized_W = Pt_W*optimized_gain;

    Pr_reference_dBm_all(trial) = 10*log10(Pr_reference_W)+30;
    Pr_optimized_dBm_all(trial) = 10*log10(Pr_optimized_W)+30;

    SNR_reference = Pr_reference_W/Noise_W;
    SNR_optimized = Pr_optimized_W/Noise_W;

    SNR_reference_dB_all(trial) = 10*log10(SNR_reference);
    SNR_optimized_dB_all(trial) = 10*log10(SNR_optimized);

    Rate_reference_all(trial) = B*log2(1+SNR_reference)/1e9;
    Rate_optimized_all(trial) = B*log2(1+SNR_optimized)/1e9;

    if mod(trial, 20) == 0
        fprintf('  trial %d/%d done\n', trial, num_trials);
    end
end

%% 3. AGGREGATE RESULTS (mean +/- std across trials)

fprintf('\n============================================\n');
fprintf(' RESULTS (mean +/- std over %d trials)\n', num_trials);
fprintf('============================================\n\n');

fprintf('--- Stage 6 reference (RIS phase = 0, LOS-matched BS) ---\n');
fprintf('Received Power = %.2f +/- %.2f dBm\n', mean(Pr_reference_dBm_all), std(Pr_reference_dBm_all));
fprintf('SNR            = %.2f +/- %.2f dB\n',  mean(SNR_reference_dB_all), std(SNR_reference_dB_all));
fprintf('Data Rate      = %.4f +/- %.4f Gbps\n', mean(Rate_reference_all), std(Rate_reference_all));

fprintf('\n--- Joint BS + RIS optimization ---\n');
fprintf('Received Power = %.2f +/- %.2f dBm\n', mean(Pr_optimized_dBm_all), std(Pr_optimized_dBm_all));
fprintf('SNR            = %.2f +/- %.2f dB\n',  mean(SNR_optimized_dB_all), std(SNR_optimized_dB_all));
fprintf('Data Rate      = %.4f +/- %.4f Gbps\n', mean(Rate_optimized_all), std(Rate_optimized_all));

Power_improvement_mean = mean(Pr_optimized_dBm_all - Pr_reference_dBm_all);
Power_improvement_std  = std(Pr_optimized_dBm_all - Pr_reference_dBm_all);
SNR_improvement_mean   = mean(SNR_optimized_dB_all - SNR_reference_dB_all);
SNR_improvement_std    = std(SNR_optimized_dB_all - SNR_reference_dB_all);
Rate_improvement_mean  = mean(Rate_optimized_all - Rate_reference_all);
Rate_improvement_std   = std(Rate_optimized_all - Rate_reference_all);

fprintf('\n--- Improvement (per-trial paired difference) ---\n');
fprintf('Power Improvement = %.2f +/- %.2f dB\n', Power_improvement_mean, Power_improvement_std);
fprintf('SNR Improvement   = %.2f +/- %.2f dB\n', SNR_improvement_mean, SNR_improvement_std);
fprintf('Rate Improvement  = %.4f +/- %.4f Gbps\n', Rate_improvement_mean, Rate_improvement_std);

fprintf('\n============================================\n');
fprintf(' STAGE 7 (MONTE CARLO) COMPLETE\n');
fprintf('============================================\n');

%% 4. Save for Stage 9

save(fullfile(resultsDir, 'stage7_montecarlo_results.mat'), ...
    'Pr_reference_dBm_all', 'Pr_optimized_dBm_all', ...
    'SNR_reference_dB_all', 'SNR_optimized_dB_all', ...
    'Rate_reference_all', 'Rate_optimized_all', ...
    'num_trials');
fprintf('Saved trial-level results to stage7_montecarlo_results.mat\n');

%% 5. Quick visualization

figure;
subplot(1,2,1);
boxplot([Rate_reference_all, Rate_optimized_all], 'Labels', {'Stage 6 (reference)', 'Stage 7 (joint opt)'});
ylabel('Data Rate (Gbps)');
title('Rate distribution across trials');
grid on;

subplot(1,2,2);
histogram(Rate_optimized_all - Rate_reference_all, 20);
xlabel('Per-trial rate improvement (Gbps)');
ylabel('Count');
title('Improvement distribution');
grid on;
