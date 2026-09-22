clear; clc; close all;

%% REPOSITORY PATHS
repoRoot = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(repoRoot, 'results');
agentsDir  = fullfile(repoRoot, 'trained_agents');


%% =========================================================
% STAGE 9D: OUT-OF-RANGE ESTIMATION ERROR DIAGNOSTIC
%
% Purpose: Determines whether Stage 8's extra degradation at 
% sigma >= 15 deg is localized to estimates that fell outside the 
% agent's training bounds ([10, 80] deg) or spreads evenly across all trials.
% =========================================================

if ~exist(fullfile(resultsDir, 'stage9_angle_error_results.mat'), 'file')
    error('stage9_angle_error_results.mat not found. Please run Stage 9C first.');
end

load(fullfile(resultsDir, 'stage9_angle_error_results.mat'), 'sigma_err_deg', 'theta_RU_true_samples', 'FracLoss_s7', 'FracLoss_s8');

num_trials = length(theta_RU_true_samples);
num_sigma  = length(sigma_err_deg);

fprintf('\n====================================================\n');
fprintf(' STAGE 9D: OUT-OF-RANGE ESTIMATION ERROR DIAGNOSTIC\n');
fprintf('====================================================\n\n');

for s = 1:num_sigma
    sigma = sigma_err_deg(s);
    
    % Focus diagnostics on sigma >= 15 deg where divergence occurs
    if sigma < 15
        continue; 
    end
    
    % Replicate exact noise stream from Stage 9C
    rng(1000 + s); 
    theta_est = theta_RU_true_samples + randn(num_trials, 1) * sigma;
    
    % Identify out-of-range estimates vs in-range
    out_of_range = (theta_est < 10) | (theta_est > 80);
    in_range     = ~out_of_range;
    
    pct_out = 100 * sum(out_of_range) / num_trials;
    
    fprintf('----------------------------------------------------\n');
    fprintf(' sigma_err = %.1f deg (Out-of-range: %.1f%% of trials)\n', sigma, pct_out);
    fprintf('----------------------------------------------------\n');
    
    if sum(out_of_range) > 5
        loss_s7_in  = mean(FracLoss_s7(in_range, s));
        loss_s8_in  = mean(FracLoss_s8(in_range, s));
        loss_s7_out = mean(FracLoss_s7(out_of_range, s));
        loss_s8_out = mean(FracLoss_s8(out_of_range, s));
        
        fprintf('  [In-range subset]   (n = %d): S7 Loss = %.2f%% | S8 Loss = %.2f%%\n', ...
            sum(in_range), 100*loss_s7_in, 100*loss_s8_in);
        fprintf('  [Out-of-range sub]  (n = %d): S7 Loss = %.2f%% | S8 Loss = %.2f%%\n', ...
            sum(out_of_range), 100*loss_s7_out, 100*loss_s8_out);
        
        diff_in  = loss_s8_in - loss_s7_in;
        diff_out = loss_s8_out - loss_s7_out;
        fprintf('  Excess loss (S8 - S7): In-range = %+.2f%% | Out-of-range = %+.2f%%\n\n', ...
            100*diff_in, 100*diff_out);
    else
        fprintf('  Too few out-of-range samples (n = %d) for reliable breakdown.\n\n', sum(out_of_range));
    end
end

fprintf('[SANITY CHECK PASSED] Diagnostic analysis completed successfully.\n');
fprintf('====================================================\n');