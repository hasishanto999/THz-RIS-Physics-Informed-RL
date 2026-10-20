clear; clc; close all;
%% =========================================================
% STAGE 9C (REVIEW-ROUND VERSION): ROBUSTNESS TO ANGLE-ESTIMATION ERROR
%
% WHAT CHANGED vs. THE ORIGINAL  (search for "REVIEW")
%   R1  N_EVAL 200 -> 2000. With 200 trials the CF mean moves by about +-0.5 Gbps between test
%       sets, which is as large as the 5-deg gains. Use 10000 if the batched actor path works.
%   R2  ONE unit-variance noise vector, scaled by sigma at every level. The sweep is therefore
%       paired across sigma and across controllers. The numbers will NOT reproduce the old
%       Table VII exactly (the old script drew a fresh noise vector per level).
%   R3  Two analytic baselines on the same draws:
%         'NF conj.'        near-field conjugate phase (knows d_BR, d_RU; ignores stuck elements)
%         'NF+fault-aware'  near-field + stuck-element-aware (the strong analytic baseline)
%   R4  Paired two-level bootstrap (trials AND seeds), 95% and Bonferroni CIs, for every learned
%       controller vs Eq.4 and vs NF+fault-aware. The per-trial t-test is still printed for
%       continuity, but it describes test-set variability only: do not quote it.
%   R5  Outputs carry a "_v2" tag so the old stage9c_results.mat / figure are not overwritten.
%
% RUN ORDER   thz_lib('selftest')  ->  this script
% REQUIRES    patched thz_lib.m, the 10-seed agent files, Statistics Toolbox (ttest, tinv)
% =========================================================

%% ===================== USER SETTINGS =====================
AGENT_TYPES   = {'2DR', '16DR', '16DS'};   % Residual and Structured agents
TRAIN_SEEDS   = 42:51;                     % Default: seeds 42 to 51
FILE_PREFIX   = 'stage8_sigma1';           % Prefix for 1-deg noise models
AGENT_DIR     = pwd;
N_EVAL        = 2000;                      % REVIEW R1: trials per sigma level (was 200)
EVAL_SEED     = 2024;
sigma_err_deg = [0 2 5 10 15 20 30];       % Angle-error std devs to sweep [deg]
N_BOOT        = 20000;                     % REVIEW R4: bootstrap resamples
ALPHA         = 0.05;

nT = numel(AGENT_TYPES);
nS = numel(TRAIN_SEEDS);
nG = numel(sigma_err_deg);
agent_file = @(type, seed) fullfile(AGENT_DIR, ...
             sprintf('%s_agent_%s_seed%d.mat', FILE_PREFIX, type, seed));

%% ===================== 1. CHECK FILES + LOAD CONFIG =====================
valid_agents = {};
fprintf('Checking available agent files for prefix "%s"...\n', FILE_PREFIX);
for t = 1:nT
    is_valid = true;
    for s = 1:nS
        f = agent_file(AGENT_TYPES{t}, TRAIN_SEEDS(s));
        if ~isfile(f)
            is_valid = false;
            break;
        end
    end
    if is_valid
        valid_agents{end+1} = AGENT_TYPES{t}; %#ok<SAGROW>
    else
        fprintf(' -> WARNING: Skipping "%s" agent (missing files for seeds with prefix %s).\n', AGENT_TYPES{t}, FILE_PREFIX);
    end
end
AGENT_TYPES = valid_agents;
nT = numel(AGENT_TYPES);

if nT == 0
    error('CRITICAL: No agent types have all seeds available for prefix "%s". Please check file names.', FILE_PREFIX);
end

cfg_t = cell(1, nT);
actors = cell(nT, nS);
for t = 1:nT
    for s = 1:nS
        S = load(agent_file(AGENT_TYPES{t}, TRAIN_SEEDS(s)), 'agent', 'THzEnv');
        actors{t, s} = getActor(S.agent);
        if s == 1, cfg_t{t} = S.THzEnv; end
    end
end
cfg = cfg_t{1};
assert(cfg.k_abs == 1.198e-3, 'k_abs must be the ITU-R value 1.198e-3');

fprintf('\n============================================\n');
fprintf(' STAGE 9C (v2): ANGLE-ERROR ROBUSTNESS\n');
fprintf(' %d sigma levels x %d trials | %d seeds x %d agent types\n', nG, N_EVAL, nS, nT);
fprintf(' (agents were trained at sigma = %.1f deg)\n', cfg.sigma_obs_deg);
fprintf('============================================\n\n');

%% ===================== 2. SWEEP =====================
rng(EVAL_SEED);
theta_true = cfg.theta_min + rand(N_EVAL, 1) * (cfg.theta_max - cfg.theta_min);  % same at every sigma
eps_unit   = randn(N_EVAL, 1);       % REVIEW R2: ONE unit-variance noise vector for ALL sigma levels

Rate_eq4 = zeros(N_EVAL, nG);                 % plane-wave closed form (Eq. 4), noisy angle
Rate_nf  = zeros(N_EVAL, nG);                 % REVIEW R3: near-field conjugate phase, noisy angle
Rate_nff = zeros(N_EVAL, nG);                 % REVIEW R3: near-field + fault-aware, noisy angle
Rate_ag  = cell(1, nT);                       % each N_EVAL x nS x nG
for t = 1:nT, Rate_ag{t} = zeros(N_EVAL, nS, nG); end

for g = 1:nG
    sigma     = sigma_err_deg(g);
    theta_est = theta_true + sigma * eps_unit;            % NOT clipped

    for i = 1:N_EVAL
        Rate_eq4(i, g) = thz_lib('evaluate', cfg, theta_true(i), thz_lib('eq4phases',     cfg, theta_est(i)));
        Rate_nf(i, g)  = thz_lib('evaluate', cfg, theta_true(i), thz_lib('nfphases',      cfg, theta_est(i)));
        Rate_nff(i, g) = thz_lib('evaluate', cfg, theta_true(i), thz_lib('nffaultphases', cfg, theta_est(i)));
    end
    for t = 1:nT
        for s = 1:nS
            Rate_ag{t}(:, s, g) = thz_lib('agentrates', cfg_t{t}, actors{t, s}, theta_true, theta_est);
        end
    end
    fprintf('sigma = %5.1f deg done | Eq.4 = %.3f | NF = %.3f | NF+fault = %.3f', sigma, ...
            mean(Rate_eq4(:, g)), mean(Rate_nf(:, g)), mean(Rate_nff(:, g)));
    for t = 1:nT
        fprintf(' | %s = %.3f', AGENT_TYPES{t}, mean(mean(Rate_ag{t}(:, :, g), 1)));
    end
    fprintf(' Gbps\n');
end

%% ===================== 3. STATISTICS =====================
nSys = nT + 3;       % Eq.4, nT learned controllers, NF conj., NF+fault-aware
sys_names = [{'Eq. 4'}, cellfun(@(x) [x ' RL'], AGENT_TYPES, 'UniformOutput', false), ...
             {'NF conj.', 'NF+fault-aware'}];

M  = zeros(nSys, nG);       % mean rate
HW = zeros(nSys, nG);       % 95% CI half-width

for g = 1:nG
    [M(1, g), HW(1, g)] = thz_lib('ci95', Rate_eq4(:, g));                    % over trials
    for t = 1:nT
        sm = squeeze(mean(Rate_ag{t}(:, :, g), 1));                            % 1 x nS (per-seed means)
        [M(t+1, g), HW(t+1, g)] = thz_lib('ci95', sm(:));                     % over seeds
    end
    [M(nT+2, g), HW(nT+2, g)] = thz_lib('ci95', Rate_nf(:, g));               % over trials
    [M(nT+3, g), HW(nT+3, g)] = thz_lib('ci95', Rate_nff(:, g));              % over trials
end

Genie_ach = zeros(N_EVAL, 1);
for i = 1:N_EVAL
    [~, inf_i] = thz_lib('evaluate', cfg, theta_true(i), zeros(cfg.N_RIS, 1));
    Genie_ach(i) = inf_i.rate_genie_ach;
end
G = mean(Genie_ach);
Pct   = 100 * M  / G;        % % of achievable optimum
PctHW = 100 * HW / G;

% Per-trial paired t-test (kept for continuity ONLY: it treats each trained agent as fixed)
P = nan(nT, nG);
for g = 1:nG
    for t = 1:nT
        d = mean(Rate_ag{t}(:, :, g), 2) - Rate_eq4(:, g);
        [~, P(t, g)] = ttest(d);
    end
end

fprintf('\n--- Mean data rate [Gbps] (95%% CI half-width in parentheses) ---\n');
fprintf('%-8s', 'sigma');
for k = 1:nSys, fprintf('%20s', sys_names{k}); end
fprintf('\n');
for g = 1:nG
    fprintf('%-8.1f', sigma_err_deg(g));
    for k = 1:nSys, fprintf('%12.3f (%5.3f)', M(k, g), HW(k, g)); end
    fprintf('\n');
end

fprintf('\n--- Rate as %% of the ACHIEVABLE OPTIMUM (%.3f Gbps) -- common reference ---\n', G);
fprintf('%-8s', 'sigma');
for k = 1:nSys, fprintf('%16s', sys_names{k}); end
fprintf('\n');
for g = 1:nG
    fprintf('%-8.1f', sigma_err_deg(g));
    for k = 1:nSys, fprintf('%16.1f', Pct(k, g)); end
    fprintf('\n');
end

% Same layout as the paper's Table VII: CF mean (% of optimum), then change relative to CF
fprintf('\n--- TABLE VII FORMAT: mean rate; change relative to Eq.4 in parentheses ---\n');
fprintf('%-8s%-18s', 'sigma', sys_names{1});
for k = 2:nSys, fprintf('%-18s', sys_names{k}); end
fprintf('\n');
for g = 1:nG
    fprintf('%-8.1f%6.2f (%5.1f)    ', sigma_err_deg(g), M(1, g), Pct(1, g));
    for k = 2:nSys
        fprintf('%6.2f (%+5.1f%%)  ', M(k, g), 100 * (M(k, g) / M(1, g) - 1));
    end
    fprintf('\n');
end

fprintf('\n--- Per-trial paired t-test p (learned minus Eq.4). DO NOT QUOTE: test-set variability only ---\n');
fprintf('%-8s', 'sigma');
for t = 1:nT, fprintf('%14s', AGENT_TYPES{t}); end
fprintf('\n');
for g = 1:nG
    fprintf('%-8.1f', sigma_err_deg(g));
    for t = 1:nT, fprintf('%14.2e', P(t, g)); end
    fprintf('\n');
end

%% ===================== 3b. REVIEW R4: SEED-LEVEL PAIRED BOOTSTRAP =====================
rng(EVAL_SEED + 1);
K_cmp = 2 * nT * nG;                 % comparisons reported below -> Bonferroni family size
Bq = cell(nT, nG);                   % learned minus Eq.4
Bn = cell(nT, nG);                   % learned minus NF+fault-aware
for g = 1:nG
    for t = 1:nT
        Bq{t, g} = seed_paired_boot(Rate_ag{t}(:, :, g), Rate_eq4(:, g), N_BOOT, ALPHA, K_cmp);
        Bn{t, g} = seed_paired_boot(Rate_ag{t}(:, :, g), Rate_nff(:, g), N_BOOT, ALPHA, K_cmp);
    end
end
print_boot_table('LEARNED minus Eq. 4 (plane-wave closed form)', Bq, AGENT_TYPES, sigma_err_deg, K_cmp);
print_boot_table('LEARNED minus NF+fault-aware analytic baseline', Bn, AGENT_TYPES, sigma_err_deg, K_cmp);
fprintf(['\nReading the tables: a CI that excludes 0 means the learned controller differs from the\n' ...
         'reference after accounting for BOTH test-trial and training-seed variability. Use the\n' ...
         'Bonferroni CI if you report the whole sweep; the 95%% CI if you report one pre-chosen sigma.\n']);

%% ===================== 4. FIGURE =====================
custom_cols = [0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19; 0.93 0.69 0.13; 0.49 0.18 0.56; 0.30 0.30 0.30];
if nSys <= size(custom_cols, 1)
    cols = custom_cols(1:nSys, :);
else
    cols = lines(nSys);
end
mk = {'o', 's', '^', 'd', 'v', '>', '<'};

fig = figure('Color', 'w', 'Position', [100 100 1100 470]);

subplot(1, 2, 1); hold on;
for k = 1:nSys
    m_idx = mod(k-1, numel(mk)) + 1;
    if k > nT + 1, ls = '--'; else, ls = '-'; end          % analytic baselines dashed
    errorbar(sigma_err_deg, M(k, :), HW(k, :), [ls mk{m_idx}], 'Color', cols(k, :), ...
             'LineWidth', 1.6, 'MarkerFaceColor', cols(k, :), 'CapSize', 8);
end
xline(cfg.sigma_obs_deg, ':k', 'training \sigma', 'LabelVerticalAlignment', 'bottom');
xlabel('Angle-estimation error std \sigma_{err} (deg)');
ylabel('Data rate (Gbps)');
legend(sys_names, 'Location', 'southwest');
title('Rate vs. angle-estimation error');
grid on; box on;

subplot(1, 2, 2); hold on;
for k = 1:nSys
    m_idx = mod(k-1, numel(mk)) + 1;
    if k > nT + 1, ls = '--'; else, ls = '-'; end
    errorbar(sigma_err_deg, Pct(k, :), PctHW(k, :), [ls mk{m_idx}], 'Color', cols(k, :), ...
             'LineWidth', 1.6, 'MarkerFaceColor', cols(k, :), 'CapSize', 8);
end
xline(cfg.sigma_obs_deg, ':k');
xlabel('Angle-estimation error std \sigma_{err} (deg)');
ylabel('Rate (% of achievable optimum)');
legend(sys_names, 'Location', 'southwest');
title('Rate relative to the achievable optimum (common reference)');
grid on; box on;

try
    exportgraphics(fig, 'fig_angle_error_sweep_v2.png', 'Resolution', 300);
catch
    saveas(fig, 'fig_angle_error_sweep_v2.png');
end

%% ===================== 5. SAVE =====================
save('stage9c_v2_results.mat', ...
     'cfg', 'AGENT_TYPES', 'TRAIN_SEEDS', 'sigma_err_deg', 'N_EVAL', 'EVAL_SEED', ...
     'theta_true', 'eps_unit', 'Rate_eq4', 'Rate_nf', 'Rate_nff', 'Rate_ag', ...
     'sys_names', 'M', 'HW', 'Pct', 'PctHW', 'G', 'P', 'Bq', 'Bn', 'N_BOOT', 'K_cmp');

fprintf('\n============================================\n');
fprintf(' STAGE 9C (v2) COMPLETE -> stage9c_v2_results.mat, fig_angle_error_sweep_v2.png\n');
fprintf('============================================\n');


%% ===================== LOCAL FUNCTIONS =====================

function out = seed_paired_boot(Rlearn, Rref, nBoot, alpha, K)
% Rlearn : M x S per-trial rates (S training seeds); Rref : M x 1 on the SAME trials/noise draws.
% Two-level bootstrap: resample trials (multinomial weights) AND seeds.
% Returns the unadjusted (1-alpha) CI and a Bonferroni CI for a family of K comparisons.
    [M, S] = size(Rlearn);
    D      = Rlearn - Rref(:);                    % M x S paired differences
    dSeed  = mean(D, 1);                          % per-seed mean difference
    bs     = zeros(nBoot, 1);
    for b = 1:nBoot
        w     = accumarray(randi(M, M, 1), 1, [M 1]);   % trial-resampling counts
        sm    = (w.' * D) / M;                          % 1 x S resampled per-seed means
        bs(b) = mean(sm(randi(S, 1, S)));               % seed resampling
    end
    bs = sort(bs);
    q  = @(a) [bs(max(1, round(nBoot * a / 2))), bs(min(nBoot, nBoot - round(nBoot * a / 2) + 1))];
    out.meanDiff         = mean(dSeed);
    out.relPct           = 100 * mean(dSeed) / mean(Rref);
    out.ci95             = q(alpha);
    out.ciBonf           = q(alpha / K);
    out.seedDiffs        = dSeed;
    out.excludesZero95   = out.ci95(1)   > 0 || out.ci95(2)   < 0;
    out.excludesZeroBonf = out.ciBonf(1) > 0 || out.ciBonf(2) < 0;
end

function print_boot_table(title_str, B, types, sig, K)
    fprintf('\n--- %s: paired seed-level bootstrap (Gbps) ---\n', title_str);
    fprintf('%-7s %-6s %9s %9s   %-19s %-19s\n', 'sigma', 'type', 'diff', 'rel.', '95% CI', sprintf('Bonf. CI (K=%d)', K));
    for g = 1:numel(sig)
        for t = 1:numel(types)
            b = B{t, g};
            fprintf('%-7.1f %-6s %+9.3f %+8.1f%%   [%+7.3f,%+7.3f]   [%+7.3f,%+7.3f]\n', ...
                    sig(g), types{t}, b.meanDiff, b.relPct, b.ci95(1), b.ci95(2), b.ciBonf(1), b.ciBonf(2));
        end
    end
end