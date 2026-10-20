clear; clc; close all;

%% =========================================================
% STAGE 9 (MULTI-SEED): THE "ULTIMATE COMPARISON" TEST
%
% Question answered:
%   When the environment is realistic and messy - near-field user, broken RIS
%   elements, and an imperfect (noisy) angle estimate - how do
%       (a) the closed-form baseline (Eq. 4),
%       (b) the 2-D RL agent (slope + curvature) and
%       (c) the 16-D RL agent (one phase per element)
%   compare in achieved data rate?
%
% FAIRNESS RULES (all enforced below)
%   * All methods are tested on the SAME true user angles (rng(EVAL_SEED)).
%   * All methods receive the SAME noisy angle estimate
%     (Eq. 4 uses it in its formula, the RL agents use it as their observation).
%   * The physical channel always uses the TRUE angle, near-field propagation
%     and the broken elements - identical for every method (thz_lib 'evaluate').
%   * RL agents are evaluated with their deterministic MEAN action.
%
% STATISTICS
%   * RL agents: mean over the 10 training seeds, 95% CI = Student-t interval
%     over the per-seed mean rates (this is the "seed-to-seed" uncertainty).
%   * Eq. 4 has no training seed, so its 95% CI is the Student-t interval over
%     the N_EVAL evaluation trials (a different kind of uncertainty; that is
%     the best one can do for a closed-form formula, and it is stated in the plot).
%   * Paired t-test per trial: (seed-averaged RL rate) - (Eq. 4 rate).
%
% CHANGES vs. THE OLD VERSION
%   - Old script tested ONE agent type with the TRUE angle in a far-field, no-fault
%     channel, 5 seeds, no confidence intervals. Now: 3 agent types, noisy angle,
%     near-field, broken elements, 10 seeds, 95% CIs, bar charts.
%   - Physics now comes from thz_lib.m (identical to training).
%   - k_abs = 1.198e-3 (ITU value) is enforced by an assertion.
%
% HOW TO RUN
%   Run stage8_multiseed_train.m first (30 agent files must exist).
%   For a smoke test, set FILE_PREFIX = 'stage8_SMOKE' and TRAIN_SEEDS = 42.
%
% OUTPUTS
%   <FILE_PREFIX>_stage9_ultimate_results.mat
%   <FILE_PREFIX>_fig_ultimate_comparison.png   <- the Eq4 / 2-D / 16-D bar chart for the paper
%   <FILE_PREFIX>_fig_extended_comparison.png   <- same + 1-D agent + individual seeds
% =========================================================

%% ===================== USER SETTINGS =====================

AGENT_TYPES = {'2DR', '16DR', '16DS'};   % residual agents (start at Eq.4). Add '1DR' if it is fixed. Old direct: '1D','2D','16D'
TRAIN_SEEDS = 42:51;            % seeds that were trained (42:51 for the 10-seed 5-deg set)
FILE_PREFIX = 'stage8_sigma1';  % 'stage8' = 5-deg agents, 'stage8_sigma1' = 1-deg agents
                                % Output files are tagged with this prefix, so scenarios never overwrite each other.
AGENT_DIR   = pwd;              % folder containing the agent .mat files
N_EVAL      = 10000;            % evaluation trials (true angles). 500 = the paper's original test set;
                                % use >= 10000 for final numbers (small test sets move CF by 0.5-1 Gbps)
EVAL_SEED   = 2024;
N_BOOT      = 20000;            % bootstrap resamples for the paired seed-level CIs (section 5b)

nT = numel(AGENT_TYPES);
nS = numel(TRAIN_SEEDS);

agent_file = @(type, seed) fullfile(AGENT_DIR, ...
             sprintf('%s_agent_%s_seed%d.mat', FILE_PREFIX, type, seed));

%% ===================== 1. CHECK FILES + LOAD CONFIG =====================

missing = {};
for t = 1:nT
    for s = 1:nS
        f = agent_file(AGENT_TYPES{t}, TRAIN_SEEDS(s));
        if ~isfile(f), missing{end+1} = f; end %#ok<SAGROW>
    end
end
if ~isempty(missing)
    error('%d agent file(s) missing, e.g.\n  %s\nRun stage8_multiseed_train.m first.', ...
          numel(missing), missing{1});
end

cfg_t = cell(1, nT);
for t = 1:nT
    tmp = load(agent_file(AGENT_TYPES{t}, TRAIN_SEEDS(1)), 'THzEnv');
    cfg_t{t} = tmp.THzEnv;
end
cfg = cfg_t{1};
for t = 2:nT
    same = isequal(cfg_t{t}.nearfield, cfg.nearfield) && ...
           isequal(cfg_t{t}.broken_idx, cfg.broken_idx) && ...
           cfg_t{t}.d_RU == cfg.d_RU && cfg_t{t}.d_BR == cfg.d_BR && ...
           cfg_t{t}.sigma_obs_deg == cfg.sigma_obs_deg;
    if ~same
        error('Agent types were trained in DIFFERENT environments - comparison would be unfair.');
    end
end
assert(cfg.k_abs == 1.198e-3, 'k_abs must be the ITU-R value 1.198e-3');

fprintf('\n============================================\n');
fprintf(' STAGE 9: ULTIMATE COMPARISON\n');
fprintf(' near-field = %d | broken elements = [%s] | angle-noise = %.1f deg | d_RU = %.3f m\n', ...
        cfg.nearfield, num2str(cfg.broken_idx), cfg.sigma_obs_deg, cfg.d_RU);
fprintf(' %d trials, %d seeds x %d agent types\n', N_EVAL, nS, nT);
fprintf('============================================\n\n');

%% ===================== 2. SHARED EVALUATION DRAWS =====================

rng(EVAL_SEED);
theta_true = cfg.theta_min + rand(N_EVAL, 1) * (cfg.theta_max - cfg.theta_min);
theta_est  = theta_true + cfg.sigma_obs_deg * randn(N_EVAL, 1);   % NOT clipped

%% ===================== 3. BASELINE (EQ. 4) + REFERENCES =====================

cfg_ideal = cfg;  cfg_ideal.nearfield = false;  cfg_ideal.broken_idx = [];

Rate_eq4          = zeros(N_EVAL, 1);   % Eq.4 fed the NOISY angle, in the messy world  <- the test
Rate_eq4_trueang  = zeros(N_EVAL, 1);   % diagnostic: Eq.4 fed the TRUE angle, messy world
Rate_eq4_ideal    = zeros(N_EVAL, 1);   % reference: Eq.4, true angle, far-field, no faults
Rate_genie_ach    = zeros(N_EVAL, 1);   % best achievable with the broken elements
Rate_genie_free   = zeros(N_EVAL, 1);   % ideal upper bound (no faults)
Rate_nf           = zeros(N_EVAL, 1);   % analytic near-field conjugate phase, NOISY angle, ignores faults
Rate_nff          = zeros(N_EVAL, 1);   % analytic near-field + fault-aware phase, NOISY angle  <- strong baseline

for i = 1:N_EVAL
    [Rate_eq4(i), inf_i] = thz_lib('evaluate', cfg, theta_true(i), ...
                                   thz_lib('eq4phases', cfg, theta_est(i)));
    Rate_genie_ach(i)  = inf_i.rate_genie_ach;
    Rate_genie_free(i) = inf_i.rate_genie_free;

    Rate_eq4_trueang(i) = thz_lib('evaluate', cfg, theta_true(i), ...
                                  thz_lib('eq4phases', cfg, theta_true(i)));

    Rate_eq4_ideal(i)   = thz_lib('evaluate', cfg_ideal, theta_true(i), ...
                                  thz_lib('eq4phases', cfg_ideal, theta_true(i)));

    % Analytic baselines that use the same noisy estimate AND the same side information
    % (d_RU, d_BR, stuck indices) that the learned controllers absorb during training.
    Rate_nf(i)  = thz_lib('evaluate', cfg, theta_true(i), thz_lib('nfphases',      cfg, theta_est(i)));
    Rate_nff(i) = thz_lib('evaluate', cfg, theta_true(i), thz_lib('nffaultphases', cfg, theta_est(i)));
end

%% ===================== 4. RL AGENTS =====================

Rate_ag = cell(1, nT);      % each: N_EVAL x nS  [Gbps]
for t = 1:nT
    Rate_ag{t} = zeros(N_EVAL, nS);
    for s = 1:nS
        S = load(agent_file(AGENT_TYPES{t}, TRAIN_SEEDS(s)), 'agent');
        actor = getActor(S.agent);
        Rate_ag{t}(:, s) = thz_lib('agentrates', cfg_t{t}, actor, theta_true, theta_est);
        fprintf('  %-3s seed %d : mean rate = %.3f Gbps (%.1f%% of achievable optimum)\n', ...
                AGENT_TYPES{t}, TRAIN_SEEDS(s), mean(Rate_ag{t}(:, s)), ...
                100 * mean(Rate_ag{t}(:, s)) / mean(Rate_genie_ach));
    end
end

%% ===================== 5. STATISTICS =====================

sys_names = [{'Eq. 4 (closed form)'}, ...
             cellfun(@(x) [x ' RL agent'], AGENT_TYPES, 'UniformOutput', false), ...
             {'NF conjugate (analytic)', 'NF + fault-aware (analytic)'}];
nSys      = nT + 3;                  % Eq.4, nT agents, 2 analytic baselines
sys_mean  = zeros(1, nSys);
sys_hw    = zeros(1, nSys);

[sys_mean(1), sys_hw(1)] = thz_lib('ci95', Rate_eq4);       % CI over trials
[sys_mean(nT+2), sys_hw(nT+2)] = thz_lib('ci95', Rate_nf);  % analytic: CI over trials
[sys_mean(nT+3), sys_hw(nT+3)] = thz_lib('ci95', Rate_nff);
seed_means = cell(1, nT);
p_vs_eq4   = nan(1, nT);
for t = 1:nT
    seed_means{t} = mean(Rate_ag{t}, 1);                    % 1 x nS
    [sys_mean(t+1), sys_hw(t+1)] = thz_lib('ci95', seed_means{t});   % CI over seeds
    d = mean(Rate_ag{t}, 2) - Rate_eq4;                     % per-trial paired difference
    [~, p_vs_eq4(t)] = ttest(d);
end

mean_ach  = mean(Rate_genie_ach);
mean_free = mean(Rate_genie_free);

fprintf('\n=====================================================================\n');
fprintf(' RESULTS  (mean data rate, Gbps, with 95%% CI)\n');
fprintf('=====================================================================\n');
fprintf('%-22s %10s %12s %14s %12s\n', 'System', 'Rate', '95% CI +/-', '% of optimum', 'p vs Eq.4');
for k = 1:nSys
    if k == 1 || k > nT + 1, pstr = '-'; else, pstr = sprintf('%.2e', p_vs_eq4(k-1)); end
    fprintf('%-22s %10.3f %12.3f %13.1f%% %12s\n', sys_names{k}, sys_mean(k), sys_hw(k), ...
            100*sys_mean(k)/mean_ach, pstr);
end
fprintf('---------------------------------------------------------------------\n');
fprintf('%-22s %10.3f   (best achievable with the broken elements)\n', 'Genie (achievable)', mean_ach);
fprintf('%-22s %10.3f   (no-fault ideal, not reachable)\n',           'Genie (no faults)',  mean_free);
fprintf('%-22s %10.3f   (Eq.4, true angle, far-field, no faults)\n',  'Ideal-env Eq. 4',   mean(Rate_eq4_ideal));
fprintf('%-22s %10.3f   (Eq.4 with TRUE angle, messy world)\n',       'Eq.4 true-angle',   mean(Rate_eq4_trueang));
fprintf('\nCI note: RL rows = seed-to-seed CI (n=%d); Eq.4 row = trial-to-trial CI (n=%d).\n', nS, N_EVAL);
fprintf('Seed spread (min / max of per-seed mean rate):\n');
for t = 1:nT
    fprintf('   %-3s : %.3f / %.3f Gbps\n', AGENT_TYPES{t}, min(seed_means{t}), max(seed_means{t}));
end

%% ===================== 5b. SEED-LEVEL PAIRED STATISTICS (review round) =====================
% The per-trial paired t-test above treats each trained agent as fixed, so it only describes
% test-set variability. The two blocks below are what to report instead:
%   (i)  paired two-level bootstrap (resample trials AND seeds) of each learned controller
%        against Eq.4 and against the analytic near-field + fault-aware baseline;
%   (ii) seed-level Welch and Mann-Whitney tests between learned controllers (the tests the
%        paper quotes for 16DS vs 2DR and 16DR vs 2DR).
rng(EVAL_SEED + 1);
K_cmp = 2 * nT;                    % number of comparisons in block (i) -> Bonferroni-widened CIs
fprintf('\n---------------------------------------------------------------------\n');
fprintf(' PAIRED SEED-LEVEL BOOTSTRAP (95%% CI, Bonferroni K = %d, %d resamples)\n', K_cmp, N_BOOT);
fprintf('---------------------------------------------------------------------\n');
boot = struct('vs_eq4', {cell(1, nT)}, 'vs_nff', {cell(1, nT)});
for t = 1:nT
    boot.vs_eq4{t} = seed_paired_boot(Rate_ag{t}, Rate_eq4, N_BOOT, 0.05, K_cmp);
    boot.vs_nff{t} = seed_paired_boot(Rate_ag{t}, Rate_nff, N_BOOT, 0.05, K_cmp);
    sA = boot.vs_eq4{t};  sB = boot.vs_nff{t};
    fprintf('%-5s vs Eq.4            : %+7.3f Gbps (%+6.1f%%)  95%% CI [%+.3f, %+.3f] | Bonf. [%+.3f, %+.3f]\n', ...
            AGENT_TYPES{t}, sA.meanDiff, sA.relPct, sA.ci95(1), sA.ci95(2), sA.ciBonf(1), sA.ciBonf(2));
    fprintf('%-5s vs NF+fault-aware  : %+7.3f Gbps (%+6.1f%%)  95%% CI [%+.3f, %+.3f] | Bonf. [%+.3f, %+.3f]\n', ...
            AGENT_TYPES{t}, sB.meanDiff, sB.relPct, sB.ci95(1), sB.ci95(2), sB.ciBonf(1), sB.ciBonf(2));
end
fprintf('\nSeed-level tests between learned controllers (n = %d seeds each):\n', nS);
for a = 1:nT-1
    for b = a+1:nT
        [~, pW] = ttest2(seed_means{b}, seed_means{a}, 'Vartype', 'unequal');   % Welch
        pMW     = ranksum(seed_means{b}, seed_means{a});                        % Mann-Whitney
        fprintf('  %-5s vs %-5s : diff = %+.3f Gbps | Welch p = %.2e | Mann-Whitney p = %.2e\n', ...
                AGENT_TYPES{b}, AGENT_TYPES{a}, mean(seed_means{b}) - mean(seed_means{a}), pW, pMW);
    end
end

%% ===================== 6. FIGURE 1: THE PAPER BAR CHART =====================

i2  = find(strcmp(AGENT_TYPES, '2DR'), 1);          % prefer the residual 2-D agent if present
if isempty(i2), i2 = find(strcmp(AGENT_TYPES, '2D'), 1); end
i16 = [];
for cand = {'16DS', '16DR', '16D'}                     % preference order for the main bar chart
    i16 = find(strcmp(AGENT_TYPES, cand{1}), 1);
    if ~isempty(i16), break; end
end
assert(~isempty(i2) && ~isempty(i16), 'AGENT_TYPES must contain ''2D'' and ''16D'' (or ''16DR'').');
idx3   = [1, 1 + i2, 1 + i16];
switch AGENT_TYPES{i16}
    case '16DS', lab16 = '16-D structured RL agent';
    case '16DR', lab16 = '16-D residual RL agent';
    otherwise,   lab16 = '16-D RL agent';
end
if strcmp(AGENT_TYPES{i2}, '2DR'), lab2 = '2-D residual RL agent'; else, lab2 = '2-D RL agent'; end
labels = {'Eq. 4 (closed form)', lab2, lab16};
cols   = [0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19];

fig1 = figure('Color', 'w', 'Position', [100 100 680 500]);
b = bar(1:3, sys_mean(idx3), 0.6);
b.FaceColor = 'flat';
b.CData = cols;
hold on;
errorbar(1:3, sys_mean(idx3), sys_hw(idx3), 'k', 'LineStyle', 'none', 'LineWidth', 1.6, 'CapSize', 14);
for k = 1:3
    text(k, sys_mean(idx3(k)) + sys_hw(idx3(k)), sprintf('%.2f', sys_mean(idx3(k))), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
end
yline(mean_ach, '--k', 'Achievable optimum (genie)', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom');
set(gca, 'XTick', 1:3, 'XTickLabel', labels, 'FontSize', 11);
ylabel('Data rate (Gbps)');
title({sprintf('Near-field + %d broken elements', numel(cfg.broken_idx)), ...
       sprintf('+ %.0f%c angle-estimation error', cfg.sigma_obs_deg, char(176))}, 'FontSize', 12);
grid on; box on;
ylim([0, 1.18 * max(sys_mean(idx3) + sys_hw(idx3))]);
annotation('textbox', [0.13 0.01 0.8 0.05], 'String', ...
    sprintf('Error bars: 95%% CI (RL: over %d training seeds; Eq. 4: over %d trials)', nS, N_EVAL), ...
    'EdgeColor', 'none', 'FontSize', 8);
save_figure(fig1, [FILE_PREFIX '_fig_ultimate_comparison.png']);

%% ===================== 7. FIGURE 2: EXTENDED (ALL AGENTS + SEEDS) =====================

fig2 = figure('Color', 'w', 'Position', [120 120 780 500]);
bar(1:nSys, sys_mean, 0.6, 'FaceColor', [0.75 0.75 0.85]);
hold on;
errorbar(1:nSys, sys_mean, sys_hw, 'k', 'LineStyle', 'none', 'LineWidth', 1.6, 'CapSize', 14);
for t = 1:nT
    plot(t + 1 + 0.25 * (rand(1, nS) - 0.5), seed_means{t}, 'k.', 'MarkerSize', 14);   % one dot per seed
end
yline(mean_ach,  '--k', 'Achievable optimum (with broken elements)', 'LabelHorizontalAlignment', 'left',  'LabelVerticalAlignment', 'bottom');
yline(mean_free, ':k',  'No-fault optimum = Eq. 4 in ideal environment', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'top');
set(gca, 'XTick', 1:nSys, 'XTickLabel', sys_names, 'FontSize', 10);
ylabel('Data rate (Gbps)');
title('All systems (dots = individual training seeds)');
grid on; box on;
save_figure(fig2, [FILE_PREFIX '_fig_extended_comparison.png']);

%% ===================== 8. SAVE =====================

save([FILE_PREFIX '_stage9_ultimate_results.mat'], ...
     'cfg', 'AGENT_TYPES', 'TRAIN_SEEDS', 'N_EVAL', 'EVAL_SEED', ...
     'theta_true', 'theta_est', ...
     'Rate_eq4', 'Rate_eq4_trueang', 'Rate_eq4_ideal', 'Rate_genie_ach', 'Rate_genie_free', ...
     'Rate_nf', 'Rate_nff', 'boot', 'N_BOOT', ...
     'Rate_ag', 'seed_means', 'sys_names', 'sys_mean', 'sys_hw', 'p_vs_eq4');

fprintf('\n============================================\n');
fprintf(' STAGE 9 COMPLETE -> %s_stage9_ultimate_results.mat, %s_fig_*.png\n', FILE_PREFIX, FILE_PREFIX);
fprintf('============================================\n');


%% ===================== LOCAL FUNCTIONS =====================

function save_figure(fig, filename)
    try
        exportgraphics(fig, filename, 'Resolution', 300);
    catch
        saveas(fig, filename);
    end
end

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