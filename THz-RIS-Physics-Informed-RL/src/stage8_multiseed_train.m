clear; clc; close all;

%% REPOSITORY PATHS
repoRoot = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(repoRoot, 'results');
agentsDir  = fullfile(repoRoot, 'trained_agents');


%% =========================================================
% STAGE 8 (MULTI-SEED): TRAIN N INDEPENDENT PPO AGENTS
%
% PPO training is stochastic (network weight init, minibatch sampling
% order, Gaussian-policy exploration noise). A single trained agent
% answers "how good is THIS trained policy?" but not
% "how much does that answer depend on which training run you happened
% to get?"
%
% This script trains N independently-seeded agents under the exact same
% environment/hyperparameters as Stage 8.
% =========================================================

TRAIN_SEEDS = [42 43 44 45 46];   % 5 independent training runs

for seed_idx = 1:numel(TRAIN_SEEDS)

    seed = TRAIN_SEEDS(seed_idx);

    fprintf('\n############################################\n');
    fprintf(' TRAINING AGENT %d/%d  (seed = %d)\n', ...
        seed_idx, numel(TRAIN_SEEDS), seed);
    fprintf('############################################\n');

    clearvars -except TRAIN_SEEDS seed_idx seed

    global THzEnv;

    %% 1. THz Environment Parameters (identical to Stage 8)

    THzEnv.fc      = 300e9;
    THzEnv.lambda  = 3e8 / THzEnv.fc;
    THzEnv.B       = 10e9;
    THzEnv.Pt_W    = 10^((20-30)/10);
    THzEnv.Noise_W = 1.380649e-23 * 290 * THzEnv.B;
    THzEnv.k_abs   = 1.198e-3;   % ITU-R P.676-12 value from Stage 2

    THzEnv.N_BS  = 8;
    THzEnv.N_RIS = 16;
    THzEnv.d_BS  = THzEnv.lambda/2;
    THzEnv.d_RIS = THzEnv.lambda/2;

    THzEnv.d_BR = 0.15;
    THzEnv.d_RU = 0.15;

    THzEnv.theta_BS = 20;
    THzEnv.theta_BR = 20;

    THzEnv.StepsPerEpisode = 50;

    %% 2. RL Observation & Action Spaces

    obsInfo = rlNumericSpec([2 1], ...
        'LowerLimit', -1, ...
        'UpperLimit', 1);

    obsInfo.Name = 'SinCos_User_Angle';

    actInfo = rlNumericSpec([1 1], ...
        'LowerLimit', -1, ...
        'UpperLimit', 1);

    actInfo.Name = 'RIS_Phase_Gradient_Slope';

    env = rlFunctionEnv(obsInfo, actInfo, @stepRIS, @resetRIS);

    %% 3. PPO Actor and Critic Networks

    % The random seed is the ONLY change between training runs
    rng(seed);

    %% Critic Network

    criticNetwork = [
        featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
        fullyConnectedLayer(64, 'Name', 'CriticFC1')
        reluLayer('Name', 'CriticRelu1')
        fullyConnectedLayer(64, 'Name', 'CriticFC2')
        reluLayer('Name', 'CriticRelu2')
        fullyConnectedLayer(1, 'Name', 'CriticOutput')
    ];

    critic = rlValueFunction(criticNetwork, obsInfo);

    %% Actor Shared Network

    obsPath = [
        featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
        fullyConnectedLayer(128, 'Name', 'ActorFC1')
        reluLayer('Name', 'ActorRelu1')
        fullyConnectedLayer(128, 'Name', 'ActorFC2')
        reluLayer('Name', 'ActorRelu2')
    ];

    %% Actor Mean Path

    meanPath = [
        fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'meanFC')
        tanhLayer('Name', 'ActionMean')
    ];

    %% Actor Standard Deviation Path

    MAX_STD = 0.5;

    stdPath = [
        fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'stdFC')
        sigmoidLayer('Name', 'stdSigmoid')
        scalingLayer('Name', 'ActionStd', ...
                     'Scale', MAX_STD)
    ];

    %% Build Actor Layer Graph

    actorNetwork = layerGraph(obsPath);

    actorNetwork = addLayers(actorNetwork, meanPath);
    actorNetwork = addLayers(actorNetwork, stdPath);

    actorNetwork = connectLayers( ...
        actorNetwork, 'ActorRelu2', 'meanFC');

    actorNetwork = connectLayers( ...
        actorNetwork, 'ActorRelu2', 'stdFC');

    %% Create Continuous Gaussian Actor

    actor = rlContinuousGaussianActor( ...
        actorNetwork, ...
        obsInfo, ...
        actInfo, ...
        'ObservationInputNames', 'state', ...
        'ActionMeanOutputNames', 'ActionMean', ...
        'ActionStandardDeviationOutputNames', 'ActionStd');

    %% 4. PPO Agent Configuration

    agentOpts = rlPPOAgentOptions( ...
        'SampleTime', 1, ...
        'DiscountFactor', 0.99, ...
        'ExperienceHorizon', 128, ...
        'ClipFactor', 0.2, ...
        'EntropyLossWeight', 0.01, ...
        'MiniBatchSize', 64, ...
        'NumEpoch', 3);

    agentOpts.ActorOptimizerOptions.LearnRate = 1e-4;
    agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;

    agent = rlPPOAgent(actor, critic, agentOpts);

    %% 5. Training Configuration

    assert( ...
        THzEnv.StepsPerEpisode == 50, ...
        'StepsPerEpisode mismatch.');

    trainOpts = rlTrainingOptions( ...
        'MaxEpisodes', 1500, ...
        'MaxStepsPerEpisode', THzEnv.StepsPerEpisode, ...
        'ScoreAveragingWindowLength', 50, ...
        'Verbose', false, ...
        'Plots', 'none', ...
        'StopTrainingCriteria', 'AverageReward', ...
        'StopTrainingValue', 0.90 * THzEnv.StepsPerEpisode);

    %% 6. Train Agent

    fprintf('Starting PPO training for seed %d...\n', seed);

    trainingStats = train( ...
        agent, ...
        env, ...
        trainOpts);

    %% 7. Save Agent Under Seed-Specific Filename

    outfile = fullfile(agentsDir, sprintf('stage8_trained_agent_seed%d.mat', seed));

    save(outfile, ...
        'agent', ...
        'THzEnv', ...
        'trainingStats', ...
        'seed');

    fprintf('Saved agent to %s\n', outfile);

end

%% =========================================================
% COMPLETION MESSAGE
% =========================================================

fprintf('\n============================================\n');
fprintf(' ALL %d SEEDS TRAINED\n', numel(TRAIN_SEEDS));
fprintf(' Files: stage8_trained_agent_seed<42..46>.mat\n');
fprintf(' Next: run stage9_multiseed_evaluation.m\n');
fprintf('============================================\n');


%% =========================================================
% LOCAL FUNCTIONS: ENVIRONMENT DYNAMICS
% =========================================================

function [InitialObservation, LoggedSignals] = resetRIS()

    theta_RU = 10 + rand() * 70;

    LoggedSignals.theta_RU = theta_RU;
    LoggedSignals.steps = 0;

    InitialObservation = [
        sind(theta_RU);
        cosd(theta_RU)
    ];

end


function [Observation, Reward, IsDone, LoggedSignals] = ...
    stepRIS(Action, LoggedSignals)

    global THzEnv;

    theta_RU = LoggedSignals.theta_RU;

    %% RIS Phase Gradient

    slope = Action(1) * pi;

    n_RIS_vec = (0:THzEnv.N_RIS-1).';

    theta_RIS_phases = n_RIS_vec * slope;

    %% Path Loss

    PL_BR_dB = ...
        20*log10(4*pi*THzEnv.d_BR/THzEnv.lambda) + ...
        4.343*THzEnv.k_abs*THzEnv.d_BR;

    PL_RU_dB = ...
        20*log10(4*pi*THzEnv.d_RU/THzEnv.lambda) + ...
        4.343*THzEnv.k_abs*THzEnv.d_RU;

    G_BR = 10^(-PL_BR_dB/10);
    G_RU = 10^(-PL_RU_dB/10);

    %% BS Steering Vector

    n_BS = (0:THzEnv.N_BS-1).';

    a_BS = exp( ...
        1i * 2*pi * THzEnv.d_BS/THzEnv.lambda * ...
        n_BS * sind(THzEnv.theta_BS));

    %% RIS Steering Vectors

    n_RIS = (0:THzEnv.N_RIS-1).';

    a_BR = exp( ...
        1i * 2*pi * THzEnv.d_RIS/THzEnv.lambda * ...
        n_RIS * sind(THzEnv.theta_BR));

    a_RU = exp( ...
        1i * 2*pi * THzEnv.d_RIS/THzEnv.lambda * ...
        n_RIS * sind(theta_RU));

    %% Channel Matrices

    H_BR = sqrt(G_BR) * (a_BR * a_BS');

    h_RU = sqrt(G_RU) * a_RU;

    %% RIS Phase Matrix

    Theta = diag(exp(1i * theta_RIS_phases));

    %% Effective BS-RIS-User Channel

    h_BS = h_RU' * Theta * H_BR;

    %% MRT Beamforming

    w = h_BS' / norm(h_BS');

    h_eff = h_BS * w;

    %% Received Power and Data Rate

    Pr_W = THzEnv.Pt_W * abs(h_eff)^2;

    SNR_linear = Pr_W / THzEnv.Noise_W;

    DataRate = THzEnv.B * log2(1 + SNR_linear);

    %% Genie-Optimal RIS

    c_genie = conj(h_RU) .* a_BR;

    theta_genie = -angle(c_genie);

    Theta_genie = diag(exp(1i * theta_genie));

    h_BS_genie = h_RU' * Theta_genie * H_BR;

    w_genie = h_BS_genie' / norm(h_BS_genie');

    h_eff_genie = h_BS_genie * w_genie;

    %% Genie Rate

    Pr_genie_W = THzEnv.Pt_W * abs(h_eff_genie)^2;

    SNR_genie = Pr_genie_W / THzEnv.Noise_W;

    Rate_genie = THzEnv.B * log2(1 + SNR_genie);

    %% Normalized Reward

    Reward = DataRate / max(Rate_genie, eps);

    %% Update Environment

    LoggedSignals.steps = LoggedSignals.steps + 1;

    LoggedSignals.theta_RU = ...
        min(80, max(10, theta_RU + randn() * 2));

    %% Next Observation

    Observation = [
        sind(LoggedSignals.theta_RU);
        cosd(LoggedSignals.theta_RU)
    ];

    %% Episode Termination

    IsDone = LoggedSignals.steps >= THzEnv.StepsPerEpisode;

end