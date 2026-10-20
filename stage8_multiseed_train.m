clear; clc; close all;

%% =========================================================
% STAGE 8 (MULTI-SEED, REALISTIC ENVIRONMENT): TRAIN 1-D / 2-D / 16-D PPO AGENTS
%
% WHAT THIS SCRIPT DOES
%   Trains 3 kinds of PPO agent x 10 independent seeds (= 30 agents) in the
%   SAME "messy" environment:
%       (1) near-field (spherical-wave) RIS links
%       (2) broken RIS elements (3 and 12 stuck at 0 phase)
%       (3) noisy angle observation (std = 5 deg) - the agent never sees the
%           true user angle; the reward is still computed from the TRUE angle
%
%   Agent types (only the ACTOR OUTPUT differs; everything else is identical):
%       '1D'  : slope                       -> 1 output  (far-field-style beam steering)
%       '2D'  : slope + curvature           -> 2 outputs (can focus a spherical wave)
%       '16D' : one phase per RIS element   -> 32 outputs = [cos x16 ; sin x16],
%               decoded with atan2  (the PHASE-WRAP FIX; see thz_lib.m)
%
% CHANGES vs. THE OLD VERSION  (search for "FIX" below)
%   FIX 1  Environment physics moved to thz_lib.m -> near-field, broken
%          elements, noisy observation (Phase 1 of the brief).
%   FIX 2  Actor output size is now dynamic (1 / 2 / 32) and 16-D uses
%          [cos, sin] + atan2 (Phase 2).
%   FIX 3  StopTrainingCriteria / StopTrainingValue REMOVED -> every agent
%          trains for the full MaxEpisodes (Phase 3). The default criterion of
%          rlTrainingOptions (AverageSteps = 500) can never fire because an
%          episode has only 50 steps, so training is never cut short.
%   FIX 4  DiscountFactor = 0 (contextual bandit: the action does not
%          influence the next state) (Phase 3).
%   FIX 5  TRAIN_SEEDS = 42:51  (10 seeds) (Phase 3).
%   FIX 6  No global variable: the config is captured by the env callbacks.
%   FIX 7  Reward is normalised by the ACHIEVABLE optimum (accounts for the
%          broken elements), so a perfect agent scores ~1.0 per step.
%
% HOW TO RUN
%   0. Put thz_lib.m in the same folder, then run   thz_lib('selftest')   once.
%   1. Set SMOKE_TEST = true and run this script (a few minutes) to check
%      that the pipeline works end-to-end. Then set it back to false.
%   2. Run this script for real. 30 agents x 1500 episodes x 50 steps: expect
%      hours, not minutes. SKIP_IF_EXISTS = true lets you stop and resume.
%   3. Then run stage9_multiseed_eval.m and stage9c_angle_error_robustness.m.
%
% OUTPUT FILES
%   stage8_agent_<TYPE>_seed<N>.mat   (agent, THzEnv = config, trainingStats, seed)
%   stage8_training_summary.mat       (final training reward, % of optimum, per type/seed)
%
% REQUIRES  Reinforcement Learning Toolbox + Deep Learning Toolbox.
% =========================================================

%% ===================== USER SETTINGS =====================

AGENT_TYPES    = { '2DR','16DR','16DS'};   % residual types: '1DR','2DR','16DR','16DS'. Existing files are skipped.
% Per-type training settings (keyed by type NAME, so AGENT_TYPES may be any subset).
% 1D / 2D are unchanged from the first run. The 16D entry uses settings aimed at its
% 32-dimensional Gaussian policy (see comments). If you tune one type, DISCLOSE it in the paper.
HP.T1D  = struct('episodes', 1500, 'entropy', 0.01,  'max_std', 0.5, 'actor_lr', 1e-4);
HP.T2D  = struct('episodes', 1500, 'entropy', 0.01,  'max_std', 0.5, 'actor_lr', 1e-4);
% Residual agents (start exactly at Eq.4): IDENTICAL settings for 1DR / 2DR / 16DR so the
% comparison isolates the number of correction parameters (1, 2 or 16).
HP.T1DR = struct('episodes', 4000, 'entropy', 0.001, 'max_std', 0.2, 'actor_lr', 3e-4);
HP.T2DR = struct('episodes', 4000, 'entropy', 0.001, 'max_std', 0.2, 'actor_lr', 3e-4);
HP.T16DS = struct('episodes', 4000, 'entropy', 0.001, 'max_std', 0.2, 'actor_lr', 3e-4);   % structured 16-D
HP.T16DR = struct('episodes', 4000, ...  % residual 16-D: starts AT Eq.4, only has to learn corrections
                  'entropy',  0.001, ...
                  'max_std',  0.2,  ...
                  'actor_lr', 3e-4);
HP.T16D = struct('episodes', 8000, ...   % first run (1500) was still under-trained
                 'entropy',  0.001, ...  % entropy bonus scales with #outputs (32) -> keeps noise high
                 'max_std',  0.2,  ...   % exploration noise on cos/sin components
                 'actor_lr', 3e-4);
TRAIN_SEEDS    = 42:51;                % 42:51 = the full 10 seeds
SKIP_IF_EXISTS = true;                 % resume-friendly
SMOKE_TEST     = false;                % true = 1 seed, 40 episodes, different filenames

% Environment realism (Phase 1)
ENV.nearfield      = true;
ENV.d_BR           = 0.15;             % BS  -> RIS [m]
ENV.d_RU           = 0.05;             % RIS -> user [m]  (inside the ~0.11 m Rayleigh distance)
ENV.broken_idx     = [3 12];
ENV.sigma_obs_deg  = 1;                % noise on the angle the agent observes [deg]  (5 = first scenario, 1 = low-noise scenario)
ENV.lowdim_cossin  = false;            % true = also send 1D/2D params as [cos,sin]+atan2
                                       % (ablation for the "stuck at 86%" problem)

% Networks / PPO
HIDDEN   = 128;

% Some MATLAB versions require DiscountFactor > 0 (error: 'Expected DiscountFactor to be positive').
% The action never changes the next state (pure contextual bandit), so any tiny value is
% equivalent to 0: the future term is 0.1% of a reward that the action cannot influence.
% USE THE SAME VALUE FOR EVERY AGENT TYPE you compare (1D/2D/16D/16DR).
DISCOUNT = 1e-3;

FILE_PREFIX = 'stage8_sigma1';   % 'stage8' = the 5-deg files. KEEP PREFIX AND ENV.sigma_obs_deg CONSISTENT!
for k = 1:numel(AGENT_TYPES)
    HP.(['T' AGENT_TYPES{k}]).episodes_full = HP.(['T' AGENT_TYPES{k}]).episodes;
end
if SMOKE_TEST
    TRAIN_SEEDS = TRAIN_SEEDS(1);
    for k = 1:numel(AGENT_TYPES)
        HP.(['T' AGENT_TYPES{k}]).episodes = min(HP.(['T' AGENT_TYPES{k}]).episodes, 40);
    end
    FILE_PREFIX = 'stage8_SMOKE';
    fprintf('*** SMOKE TEST: 1 seed, %d episodes. Files are prefixed "%s". ***\n', 40, FILE_PREFIX);
end

fprintf('\n>>> SCENARIO: angle noise = %.1f deg | file prefix = %s | types = %s | seeds = %d..%d <<<\n', ...
        ENV.sigma_obs_deg, FILE_PREFIX, strjoin(AGENT_TYPES, ','), TRAIN_SEEDS(1), TRAIN_SEEDS(end));

nT = numel(AGENT_TYPES);
nS = numel(TRAIN_SEEDS);
pct_final = nan(nT, nS);    % last-100-episode training reward as % of the achievable optimum

%% ===================== TRAINING LOOP =====================

for t = 1:nT

    type  = AGENT_TYPES{t};
    hp    = HP.(['T' type]);
    nEp   = hp.episodes;

    for s = 1:nS

        seed    = TRAIN_SEEDS(s);
        outfile = sprintf('%s_agent_%s_seed%d.mat', FILE_PREFIX, type, seed);

        fprintf('\n############################################\n');
        fprintf(' AGENT TYPE %-3s | seed %d  (%d/%d of this type) | %d episodes\n', ...
                type, seed, s, nS, nEp);
        fprintf('############################################\n');

        % ---------- resume support ----------
        if SKIP_IF_EXISTS && isfile(outfile)
            old = load(outfile, 'trainingStats');
            pct_final(t, s) = final_pct(old.trainingStats, 50);
            fprintf('  %s already exists -> skipped (final = %.1f%% of optimum)\n', outfile, pct_final(t, s));
            continue;
        end

        %% 1. Config (single source of truth = thz_lib) -------------------
        cfg = thz_lib('defaultcfg');
        cfg.agent_type    = type;
        cfg.nearfield     = ENV.nearfield;
        cfg.d_BR          = ENV.d_BR;
        cfg.d_RU          = ENV.d_RU;
        cfg.broken_idx    = ENV.broken_idx;
        cfg.sigma_obs_deg = ENV.sigma_obs_deg;
        cfg.lowdim_cossin = ENV.lowdim_cossin;

        assert(cfg.k_abs == 1.198e-3, 'k_abs must be the ITU-R value 1.198e-3');

        %% 2. Observation / action specs (FIX 2: dynamic action size) -----
        nAct = thz_lib('actdim', cfg);

        obsInfo = rlNumericSpec([2 1], 'LowerLimit', -1, 'UpperLimit', 1);
        obsInfo.Name = 'SinCos_Noisy_User_Angle';

        actInfo = rlNumericSpec([nAct 1], 'LowerLimit', -1, 'UpperLimit', 1);
        actInfo.Name = sprintf('RIS_Action_%s', type);

        % FIX 6: callbacks capture cfg (no globals)
        env = rlFunctionEnv(obsInfo, actInfo, ...
                            @(a, ls) stepRIS(a, ls, cfg), ...
                            @() resetRIS(cfg));

        %% 3. Networks (the ONLY thing that differs between runs is the seed)
        rng(seed);

        criticNetwork = [
            featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
            fullyConnectedLayer(64, 'Name', 'CriticFC1')
            reluLayer('Name', 'CriticRelu1')
            fullyConnectedLayer(64, 'Name', 'CriticFC2')
            reluLayer('Name', 'CriticRelu2')
            fullyConnectedLayer(1, 'Name', 'CriticOutput')
        ];
        critic = rlValueFunction(criticNetwork, obsInfo);

        obsPath = [
            featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
            fullyConnectedLayer(HIDDEN, 'Name', 'ActorFC1')
            reluLayer('Name', 'ActorRelu1')
            fullyConnectedLayer(HIDDEN, 'Name', 'ActorFC2')
            reluLayer('Name', 'ActorRelu2')
        ];

        if any(strcmp(type, {'1DR', '2DR', '16DR', '16DS'}))
            % zero-initialised mean layer -> initial policy mean = 0 -> phases = Eq.4 exactly
            meanFC = fullyConnectedLayer(nAct, 'Name', 'meanFC', ...
                        'Weights', zeros(nAct, HIDDEN), 'Bias', zeros(nAct, 1));
        else
            meanFC = fullyConnectedLayer(nAct, 'Name', 'meanFC');
        end
        meanPath = [
            meanFC
            tanhLayer('Name', 'ActionMean')
        ];

        stdPath = [
            fullyConnectedLayer(nAct, 'Name', 'stdFC')
            sigmoidLayer('Name', 'stdSigmoid')
            scalingLayer('Name', 'ActionStd', 'Scale', hp.max_std)
        ];

        actorNetwork = layerGraph(obsPath);
        actorNetwork = addLayers(actorNetwork, meanPath);
        actorNetwork = addLayers(actorNetwork, stdPath);
        actorNetwork = connectLayers(actorNetwork, 'ActorRelu2', 'meanFC');
        actorNetwork = connectLayers(actorNetwork, 'ActorRelu2', 'stdFC');

        actor = rlContinuousGaussianActor(actorNetwork, obsInfo, actInfo, ...
                    'ObservationInputNames',              'state', ...
                    'ActionMeanOutputNames',              'ActionMean', ...
                    'ActionStandardDeviationOutputNames', 'ActionStd');

        %% 4. PPO agent options (FIX 4: DiscountFactor = 0)
        agentOpts = rlPPOAgentOptions( ...
            'SampleTime',        1, ...
            'DiscountFactor',    DISCOUNT, ...   % ~0 = contextual bandit (MATLAB rejects exactly 0)
            'ExperienceHorizon', 128, ...
            'ClipFactor',        0.2, ...
            'EntropyLossWeight', hp.entropy, ...
            'MiniBatchSize',     64, ...
            'NumEpoch',          3);
        agentOpts.ActorOptimizerOptions.LearnRate  = hp.actor_lr;
        agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;

        agent = rlPPOAgent(actor, critic, agentOpts);

        %% 5. Training options (FIX 3: NO early stopping)
        trainOpts = rlTrainingOptions( ...
            'MaxEpisodes',                nEp, ...
            'MaxStepsPerEpisode',         cfg.StepsPerEpisode, ...
            'ScoreAveragingWindowLength', 50, ...
            'Verbose',                    false, ...
            'Plots',                      'none');
        % (StopTrainingCriteria / StopTrainingValue deliberately NOT set.)

        %% 6. Train
        tic;
        trainingStats = train(agent, env, trainOpts);
        elapsed = toc;

        nDone = numel(trainingStats.EpisodeReward);
        if nDone < nEp
            warning('Agent %s seed %d stopped after %d/%d episodes!', type, seed, nDone, nEp);
        end

        pct_final(t, s) = final_pct(trainingStats, cfg.StepsPerEpisode);
        fprintf('  done: %d episodes in %.1f min | last-100-episode reward = %.1f%% of achievable optimum\n', ...
                nDone, elapsed/60, pct_final(t, s));

        %% 7. Save (variable name THzEnv kept for compatibility with the eval scripts)
        THzEnv     = cfg;
        agent_type = type;
        save(outfile, 'agent', 'THzEnv', 'trainingStats', 'seed', 'agent_type');
        fprintf('  saved -> %s\n', outfile);
    end
end

%% ===================== SUMMARY =====================

fprintf('\n============================================\n');
fprintf(' TRAINING SUMMARY  (last-100-episode reward, %% of achievable optimum)\n');
fprintf(' NOTE: includes exploration noise, so it is slightly BELOW the deterministic\n');
fprintf('       evaluation score. Look for seeds that lag far behind their siblings.\n');
fprintf('============================================\n');
fprintf('%-6s', 'type');
fprintf('%8d', TRAIN_SEEDS);
fprintf('   | mean\n');
for t = 1:nT
    fprintf('%-6s', AGENT_TYPES{t});
    fprintf('%8.1f', pct_final(t, :));
    fprintf('   | %.1f\n', mean(pct_final(t, :), 'omitnan'));
end

save(sprintf('%s_training_summary.mat', FILE_PREFIX), ...
     'pct_final', 'AGENT_TYPES', 'TRAIN_SEEDS', 'HP', 'ENV');

fprintf('\nNext: run stage9_multiseed_eval.m  (set FILE_PREFIX = ''%s'' there).\n', FILE_PREFIX);


%% =========================================================
% LOCAL FUNCTIONS
% =========================================================

function p = final_pct(trainingStats, stepsPerEpisode)
% mean of the last 100 episode rewards, as % of the per-step optimum (= 1.0)
    r = trainingStats.EpisodeReward;
    r = r(max(1, end-99):end);
    p = 100 * mean(r) / stepsPerEpisode;
end

function [obs, theta_obs] = make_observation(theta_true_deg, cfg)
% What the AGENT sees: the true angle corrupted by estimation noise.
    theta_obs = theta_true_deg + cfg.sigma_obs_deg * randn();
    obs = [sind(theta_obs); cosd(theta_obs)];
end

function [InitialObservation, LoggedSignals] = resetRIS(cfg)
    theta_true = cfg.theta_min + rand() * (cfg.theta_max - cfg.theta_min);
    LoggedSignals.theta_RU = theta_true;      % TRUE angle (hidden from the agent)
    LoggedSignals.steps    = 0;
    [InitialObservation, LoggedSignals.theta_obs] = make_observation(theta_true, cfg);   % theta_obs = what the agent sees
end

function [Observation, Reward, IsDone, LoggedSignals] = stepRIS(Action, LoggedSignals, cfg)

    if iscell(Action), Action = Action{1}; end
    theta_true = LoggedSignals.theta_RU;

    % RL action -> RIS phases (1-D / 2-D / 16-D via [cos,sin] + atan2)
    % (theta_obs = the noisy angle the agent saw when it chose this action; only '16DR' uses it)
    phases = thz_lib('action2phases', cfg, Action(:), LoggedSignals.theta_obs);

    % Physical link: TRUE angle, near-field, broken elements forced to 0 phase
    [Rate, info] = thz_lib('evaluate', cfg, theta_true, phases);

    % FIX 7: normalise by the best rate ACHIEVABLE with the broken elements
    Reward = Rate / max(info.rate_genie_ach, eps);

    % User moves (random walk, clipped to the training range)
    LoggedSignals.steps = LoggedSignals.steps + 1;
    theta_next = theta_true + cfg.walk_std_deg * randn();
    LoggedSignals.theta_RU = min(cfg.theta_max, max(cfg.theta_min, theta_next));

    % Next observation = NOISY estimate of the new angle
    [Observation, LoggedSignals.theta_obs] = make_observation(LoggedSignals.theta_RU, cfg);

    IsDone = LoggedSignals.steps >= cfg.StepsPerEpisode;
end
