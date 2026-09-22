clear; clc; close all;

%% REPOSITORY PATHS
repoRoot = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(repoRoot, 'results');
agentsDir  = fullfile(repoRoot, 'trained_agents');


%% =========================================================
% STAGE 8 (DOMAIN-RANDOMIZED): TRAIN PPO WITH NOISY OBSERVATIONS
%
% Motivation (from Stage 9D diagnostic): the vanilla agent (trained with
% PERFECT angle observations) shows a large, out-of-distribution-shaped
% robustness gap vs. the closed form specifically when the noisy angle
% ESTIMATE at test time falls outside the training range [10,80] deg -
% the in-range excess loss is ~0-1%, the out-of-range excess loss is
% 9-14%, consistently across sigma=15/20/30 deg.
%
% This script closes that gap at the source: during training, the agent
% is still deployed in the SAME physical scenario (true theta_RU ~
% U(10,80), the real angle the environment simulates), but its
% OBSERVATION is corrupted with estimation noise of randomly-varying
% strength every episode - exactly mirroring what Stage 9C does at test
% time. This means the observation the actor sees during training
% routinely falls outside [10,80] too, so "out of range" is no longer
% out of DISTRIBUTION for the trained policy.
%
% The physical channel and reward are still computed from the TRUE
% angle (reality doesn't care about the estimate) - only the actor's
% INPUT changes. This is the standard domain-randomization recipe:
% randomize the nuisance variable (here, observation noise strength)
% during training so the deployed policy is not surprised by it.
%
% Identical to stage8_ppo_training.m otherwise (same architecture,
% hyperparameters, MaxEpisodes=1500, rng(42)) so the comparison to the
% vanilla agent isolates this one change.
% =========================================================

global THzEnv;

%% 1. THz Environment Parameters (identical to Stage 8)

THzEnv.fc     = 300e9;
THzEnv.lambda = 3e8 / THzEnv.fc;
THzEnv.B      = 10e9;
THzEnv.Pt_W   = 10^((20-30)/10);
THzEnv.Noise_W = 1.380649e-23 * 290 * THzEnv.B;
THzEnv.k_abs  = 1.198e-3;   % ITU-R P.676-12 value from Stage 2

THzEnv.N_BS  = 8;
THzEnv.N_RIS = 16;
THzEnv.d_BS  = THzEnv.lambda/2;
THzEnv.d_RIS = THzEnv.lambda/2;

THzEnv.d_BR = 0.15;
THzEnv.d_RU = 0.15;

THzEnv.theta_BS = 20;
THzEnv.theta_BR = 20;

THzEnv.StepsPerEpisode = 50;

% ---- Domain randomization setting ----
% Each episode draws its own noise strength sigma_train ~ U(0, SIGMA_MAX),
% so training sees the full spectrum from perfect knowledge up to heavy
% error, matching the sigma range Stage 9C stress-tests (0-30 deg).
% SIGMA_MAX=25 covers the region (15-30 deg) where the OOD gap appeared,
% without spending most of training on scenarios far outside anything
% Stage 9C evaluates.
THzEnv.SIGMA_MAX_TRAIN = 25;   % deg

%% 2. RL Observation & Action Spaces (unchanged)

obsInfo = rlNumericSpec([2 1], 'LowerLimit', -1, 'UpperLimit', 1);
obsInfo.Name = 'SinCos_User_Angle_Estimate';   % renamed: this is now an ESTIMATE, not ground truth

actInfo = rlNumericSpec([1 1], 'LowerLimit', -1, 'UpperLimit', 1);
actInfo.Name = 'RIS_Phase_Gradient_Slope';

env = rlFunctionEnv(obsInfo, actInfo, @stepRIS, @resetRIS);

%% 3. PPO Actor and Critic Networks (identical architecture)

rng(42);   % same seed as the canonical vanilla agent, for a controlled comparison

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
    fullyConnectedLayer(128, 'Name', 'ActorFC1')
    reluLayer('Name', 'ActorRelu1')
    fullyConnectedLayer(128, 'Name', 'ActorFC2')
    reluLayer('Name', 'ActorRelu2')
];

meanPath = [
    fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'meanFC')
    tanhLayer('Name', 'ActionMean')
];

MAX_STD = 0.5;
stdPath = [
    fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'stdFC')
    sigmoidLayer('Name', 'stdSigmoid')
    scalingLayer('Name', 'ActionStd', 'Scale', MAX_STD)
];

actorNetwork = layerGraph(obsPath);
actorNetwork = addLayers(actorNetwork, meanPath);
actorNetwork = addLayers(actorNetwork, stdPath);
actorNetwork = connectLayers(actorNetwork, 'ActorRelu2', 'meanFC');
actorNetwork = connectLayers(actorNetwork, 'ActorRelu2', 'stdFC');

actor = rlContinuousGaussianActor(actorNetwork, obsInfo, actInfo, ...
    'ObservationInputNames', 'state', ...
    'ActionMeanOutputNames', 'ActionMean', ...
    'ActionStandardDeviationOutputNames', 'ActionStd');

%% 4. PPO Agent Configuration (identical hyperparameters)

agentOpts = rlPPOAgentOptions(...
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

%% 5. Training Configuration (identical budget to the fixed vanilla agent)

assert(THzEnv.StepsPerEpisode == 50, ...
    'THzEnv.StepsPerEpisode must match MaxStepsPerEpisode below.');

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 1500, ...
    'MaxStepsPerEpisode', THzEnv.StepsPerEpisode, ...
    'ScoreAveragingWindowLength', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 0.90 * THzEnv.StepsPerEpisode);

%% 6. Train the Agent

fprintf('\n============================================\n');
fprintf(' STAGE 8 (DOMAIN-RANDOMIZED): TRAINING PPO AGENT\n');
fprintf(' Observation noise ~ U(0, %.0f) deg, resampled every episode\n', THzEnv.SIGMA_MAX_TRAIN);
fprintf('============================================\n');

trainingStats = train(agent, env, trainOpts);

fprintf('\n============================================\n');
fprintf(' TRAINING COMPLETE\n');
fprintf('============================================\n');

%% 7. Save under its own filename - does NOT overwrite the vanilla agent

save(fullfile(agentsDir, 'stage8_trained_agent_domainrand.mat'), 'agent', 'THzEnv', 'trainingStats');
fprintf('Saved agent to stage8_trained_agent_domainrand.mat\n');
fprintf(['To evaluate: point stage9_angle_error_robustness.m''s load() at this file (or copy it to\n' ...
         ' stage8_trained_agent.mat temporarily), rerun Stage 9C then Stage 9D, and compare the new\n' ...
         ' out-of-range excess-loss numbers against the vanilla agent''s (13.88%%/9.41%%/11.49%% at\n' ...
         ' sigma=15/20/30 deg) to see whether domain randomization closed the gap.\n']);

%% =========================================================
% LOCAL FUNCTIONS: ENVIRONMENT DYNAMICS (with observation noise)
% =========================================================

function [InitialObservation, LoggedSignals] = resetRIS()
    global THzEnv;

    theta_RU_true = 10 + rand() * 70;   % TRUE physical angle - scenario unchanged

    % Domain randomization: this episode's observation-noise strength,
    % re-sampled fresh every episode so training covers the full range
    % of estimation quality Stage 9C evaluates.
    sigma_train = rand() * THzEnv.SIGMA_MAX_TRAIN;

    LoggedSignals.theta_RU_true = theta_RU_true;
    LoggedSignals.sigma_train   = sigma_train;
    LoggedSignals.steps = 0;

    theta_obs = theta_RU_true + randn() * sigma_train;   % noisy ESTIMATE the actor sees
    InitialObservation = [sind(theta_obs); cosd(theta_obs)];
end

function [Observation, Reward, IsDone, LoggedSignals] = stepRIS(Action, LoggedSignals)
    global THzEnv;

    theta_RU_true = LoggedSignals.theta_RU_true;
    sigma_train   = LoggedSignals.sigma_train;

    % Agent's action was computed from the NOISY observation it received
    % last step - applied to the TRUE physical channel, exactly matching
    % how Stage 9C evaluates estimation-error robustness.
    slope = Action(1) * pi;
    n_RIS_vec = (0:THzEnv.N_RIS-1).';
    theta_RIS_phases = n_RIS_vec * slope;

    % ---- TRUE physical channel (reality uses the TRUE angle) ----
    PL_BR_dB = 20*log10(4*pi*THzEnv.d_BR/THzEnv.lambda) + 4.343*THzEnv.k_abs*THzEnv.d_BR;
    PL_RU_dB = 20*log10(4*pi*THzEnv.d_RU/THzEnv.lambda) + 4.343*THzEnv.k_abs*THzEnv.d_RU;

    G_BR = 10^(-PL_BR_dB/10);
    G_RU = 10^(-PL_RU_dB/10);

    n_BS  = (0:THzEnv.N_BS-1).';
    a_BS  = exp(1i*2*pi*THzEnv.d_BS/THzEnv.lambda * n_BS*sind(THzEnv.theta_BS));

    n_RIS = (0:THzEnv.N_RIS-1).';
    a_BR  = exp(1i*2*pi*THzEnv.d_RIS/THzEnv.lambda * n_RIS*sind(THzEnv.theta_BR));
    a_RU  = exp(1i*2*pi*THzEnv.d_RIS/THzEnv.lambda * n_RIS*sind(theta_RU_true));

    H_BR = sqrt(G_BR) * (a_BR * a_BS');
    h_RU = sqrt(G_RU) * a_RU;

    Theta = diag(exp(1i*theta_RIS_phases));
    h_BS  = h_RU' * Theta * H_BR;
    w     = h_BS' / norm(h_BS');
    h_eff = h_BS * w;

    Pr_W = THzEnv.Pt_W * abs(h_eff)^2;
    SNR_linear = Pr_W / THzEnv.Noise_W;
    DataRate = THzEnv.B * log2(1 + SNR_linear);

    % ---- Genie-optimal rate: computed from the TRUE angle (the actual
    %      achievable optimum for THIS physical channel), same as
    %      Stage 8's original design - reward is normalized to what's
    %      truly achievable, not degraded by the observation noise ----
    c_genie = conj(h_RU) .* a_BR;
    theta_genie = -angle(c_genie);
    Theta_genie = diag(exp(1i*theta_genie));

    h_BS_genie = h_RU' * Theta_genie * H_BR;
    w_genie    = h_BS_genie' / norm(h_BS_genie');
    h_eff_genie = h_BS_genie * w_genie;

    Pr_genie_W = THzEnv.Pt_W * abs(h_eff_genie)^2;
    SNR_genie  = Pr_genie_W / THzEnv.Noise_W;
    Rate_genie = THzEnv.B * log2(1 + SNR_genie);

    Reward = DataRate / max(Rate_genie, eps);

    % ---- Advance TRUE angle (same random walk as before) ----
    LoggedSignals.steps = LoggedSignals.steps + 1;
    theta_RU_true_next = min(80, max(10, theta_RU_true + randn() * 2));
    LoggedSignals.theta_RU_true = theta_RU_true_next;

    % ---- Next observation: fresh noise draw at this episode's sigma_train ----
    theta_obs_next = theta_RU_true_next + randn() * sigma_train;
    Observation = [sind(theta_obs_next); cosd(theta_obs_next)];
    IsDone = LoggedSignals.steps >= 50;
end