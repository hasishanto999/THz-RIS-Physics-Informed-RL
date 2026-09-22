clear;
clc;
close all;

%% REPOSITORY PATHS
repoRoot = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(repoRoot, 'results');
agentsDir  = fullfile(repoRoot, 'trained_agents');


%% =========================================================
% STAGE 8: PPO DEEP REINFORCEMENT LEARNING FOR RIS (FIXED v2)
%
% Changes vs. previous version:
%   1. Distances / antenna count now match Stage 7 (0.15 m / 0.15 m,
%      N_BS = 8) instead of 10 m / 5 m, N_BS = 4. The old geometry gave
%      SNR ~ -51 dB even at the *optimal* RIS phase, so the reward
%      (DataRate/120) was ~0 everywhere -> flat gradient, not primarily
%      an architecture problem.
%   2. Removed the extra per-element antenna gain terms (G_BS_elem,
%      G_RIS_elem, G_user) that weren't present in Stage 6/7 - keeping
%      Stage 8's channel model identical to Stage 7's so the Stage 9
%      comparison is apples-to-apples.
%   3. Reward is now normalized by an analytically computed
%      genie-optimal rate (closed form, derived below) instead of a
%      hardcoded guess. This adapts automatically and gives a dense,
%      well-scaled signal instead of a near-zero constant.
%   4. State is now [sin(theta_RU), cos(theta_RU)] instead of a raw
%      normalized scalar, per your Option 1.
%   5. Kept the channel LOS-only (no Rician NLOS) on purpose: the state
%      only encodes the angle, not full CSI, so the state->optimal
%      action map is only well-defined if the channel is deterministic
%      given the angle. Adding NLOS scatter here without feeding CSI
%      into the state would make the problem ill-posed. State this
%      simplification explicitly in the paper.
% =========================================================

global THzEnv;

%% 1. THz Environment Parameters (ALIGNED WITH STAGE 7)

THzEnv.fc     = 300e9;
THzEnv.lambda = 3e8 / THzEnv.fc;
THzEnv.B      = 10e9;
THzEnv.Pt_W   = 10^((20-30)/10);
THzEnv.Noise_W = 1.380649e-23 * 290 * THzEnv.B;
THzEnv.k_abs  = 1.198e-3;   % ITU-R P.676-12 line-by-line value from Stage 2
                             % (fc=300 GHz, standard atmosphere), replacing the
                             % earlier 0.005 Np/m placeholder. This only rescales
                             % G_BR/G_RU by a fixed <0.005 dB/link amount and does
                             % not change theta_genie (Eq. 23-24), so it does not
                             % require re-deriving the action-space reduction.

THzEnv.N_BS  = 8;        % matched to Stage 7 (was 4)
THzEnv.N_RIS = 16;
THzEnv.d_BS  = THzEnv.lambda/2;
THzEnv.d_RIS = THzEnv.lambda/2;

THzEnv.d_BR = 0.15;      % matched to Stage 7 (was 10.0 m)
THzEnv.d_RU = 0.15;      % matched to Stage 7 (was 5.0 m)

THzEnv.theta_BS = 20;
THzEnv.theta_BR = 20;

THzEnv.StepsPerEpisode = 50;   % must match MaxStepsPerEpisode below -
                                 % used to keep cumulative episode reward
                                 % in [0,1] instead of [0, StepsPerEpisode]

%% 2. RL Observation & Action Spaces

% State: [sin(theta_RU), cos(theta_RU)] - cyclic encoding, no
% discontinuity at the wrap-around point, maps far more naturally
% onto what the network has to compute than a raw scalar.
obsInfo = rlNumericSpec([2 1], 'LowerLimit', -1, 'UpperLimit', 1);
obsInfo.Name = 'SinCos_User_Angle';

% Action: single scalar - the RIS phase-gradient slope, normalized to
% [-1,1]. For this LOS-only rank-1 channel, the true optimal phase at
% element n is EXACTLY linear in n:
%   theta_genie_n = n * 2*pi*(d_RIS/lambda)*(sind(theta_RU) - sind(theta_BS_or_RIS))
% i.e. one real degree of freedom, not 16 independent ones. Letting the
% agent output the whole 16-dim phase vector directly makes the actor
% solve a much harder, under-constrained credit-assignment problem than
% the physics requires - this is very likely why training collapsed to a
% flat ~7%-of-optimal policy instead of climbing. Outputting just the
% slope is a lossless reparameterization for this channel model, not an
% approximation.
actInfo = rlNumericSpec([1 1], 'LowerLimit', -1, 'UpperLimit', 1);
actInfo.Name = 'RIS_Phase_Gradient_Slope';

env = rlFunctionEnv(obsInfo, actInfo, @stepRIS, @resetRIS);

%% 3. PPO Actor and Critic Networks

% Fixed seed for reproducibility: PPO training is stochastic (network
% weight initialization, minibatch sampling order, and Gaussian-policy
% exploration noise all differ run-to-run otherwise), so without this,
% every downstream result in Stage 9B/9C/9D is silently tied to whichever
% training run happened to be most recent. Document this seed value in
% the paper's Methodology section (Sec. IV-D) so results are reproducible.
rng(42);

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

% Std is now hard-bounded to (0, MAX_STD) via sigmoid+scaling instead of a
% plain softplus, which has no ceiling. The diagnostic showed std had
% reached ~120 - a bounded output makes that structurally impossible
% regardless of how the entropy/reward balance behaves, rather than
% relying solely on tuning EntropyLossWeight to keep it in check.
MAX_STD = 0.5;   % generous upper bound for a [-1,1]-scaled action; typical
                  % converged continuous-control policies run well under 1

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

%% 4. PPO Agent Configuration

% Entropy weight lowered (0.05 -> 0.02 -> 0.01 across iterations) as a
% safety margin. This is now a backstop, not the primary fix - the
% primary fixes are restoring the reward to O(1) per step and
% hard-bounding std via sigmoid (both above), which address why the
% entropy bonus was dominating the loss and blowing up std in the first
% place.
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

%% 5. Training Configuration

assert(THzEnv.StepsPerEpisode == 50, ...
    'THzEnv.StepsPerEpisode must match MaxStepsPerEpisode below, or the reward normalization and StopTrainingValue go out of sync again.');

% raised from 500 - training had not converged by ep. 500 (avg reward
% still climbing, stop criterion of 45 not yet met); revisit if it
% still hasn't plateaued by 1500
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 1500, ...       
    'MaxStepsPerEpisode', THzEnv.StepsPerEpisode, ...
    'ScoreAveragingWindowLength', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 0.90 * THzEnv.StepsPerEpisode);
% Reward is no longer divided by StepsPerEpisode, so the cumulative
% episode reward now ranges up to ~StepsPerEpisode (50) at genie-optimal
% every step. 90% of that is 45, not 0.90 - this threshold reflects that.

%% 6. Train the Agent

fprintf('\n============================================\n');
fprintf(' STAGE 8: TRAINING PPO AGENT (FIXED)\n');
fprintf('============================================\n');

trainingStats = train(agent, env, trainOpts);

fprintf('\n============================================\n');
fprintf(' TRAINING COMPLETE\n');
fprintf('============================================\n');

%% 7. Save the trained agent + environment so diagnostics work after clear/restart
save(fullfile(agentsDir, 'stage8_trained_agent.mat'), 'agent', 'THzEnv', 'trainingStats');
fprintf('Saved agent + environment to stage8_trained_agent.mat\n');

%% =========================================================
% LOCAL FUNCTIONS: ENVIRONMENT DYNAMICS
% =========================================================

function [InitialObservation, LoggedSignals] = resetRIS()
    theta_RU = 10 + rand() * 70;

    LoggedSignals.theta_RU = theta_RU;
    LoggedSignals.steps = 0;

    InitialObservation = [sind(theta_RU); cosd(theta_RU)];
end

function [Observation, Reward, IsDone, LoggedSignals] = stepRIS(Action, LoggedSignals)
    global THzEnv;

    theta_RU = LoggedSignals.theta_RU;

    % Agent outputs one scalar (normalized slope in [-1,1]); expand to the
    % full 16-element linear phase ramp. Scale by pi: the true slope
    % magnitude for this geometry never exceeds ~pi (max ~2.02 rad across
    % theta_RU in [10,80] deg with theta_BR=20 deg fixed), so this bound
    % covers the needed range without the agent needing to saturate.
    slope = Action(1) * pi;
    n_RIS_vec = (0:THzEnv.N_RIS-1).';
    theta_RIS_phases = n_RIS_vec * slope;

    % ---- Channel (pure LOS, matches Stage 7's rank-1 structure) ----
    PL_BR_dB = 20*log10(4*pi*THzEnv.d_BR/THzEnv.lambda) + 4.343*THzEnv.k_abs*THzEnv.d_BR;
    PL_RU_dB = 20*log10(4*pi*THzEnv.d_RU/THzEnv.lambda) + 4.343*THzEnv.k_abs*THzEnv.d_RU;

    G_BR = 10^(-PL_BR_dB/10);
    G_RU = 10^(-PL_RU_dB/10);

    n_BS  = (0:THzEnv.N_BS-1).';
    a_BS  = exp(1i*2*pi*THzEnv.d_BS/THzEnv.lambda * n_BS*sind(THzEnv.theta_BS));

    n_RIS = (0:THzEnv.N_RIS-1).';
    a_BR  = exp(1i*2*pi*THzEnv.d_RIS/THzEnv.lambda * n_RIS*sind(THzEnv.theta_BR));
    a_RU  = exp(1i*2*pi*THzEnv.d_RIS/THzEnv.lambda * n_RIS*sind(theta_RU));

    H_BR = sqrt(G_BR) * (a_BR * a_BS');   % rank-1
    h_RU = sqrt(G_RU) * a_RU;

    % ---- Agent's action ----
    Theta = diag(exp(1i*theta_RIS_phases));
    h_BS  = h_RU' * Theta * H_BR;
    w     = h_BS' / norm(h_BS');
    h_eff = h_BS * w;

    Pr_W = THzEnv.Pt_W * abs(h_eff)^2;
    SNR_linear = Pr_W / THzEnv.Noise_W;
    DataRate = THzEnv.B * log2(1 + SNR_linear);

    % ---- Genie-optimal rate for THIS channel realization, closed form ----
    % Because H_BR is rank-1 (H_BR = sqrt(G_BR) a_BR a_BS'), the BS-side
    % optimal MRT direction is ALWAYS a_BS regardless of RIS phase, so the
    % RIS phase that maximizes |h_RU' * Theta * a_BR| is optimal jointly,
    % with no alternating iterations needed:
    %   theta_genie_n = -angle( conj(h_RU_n) * a_BR_n )
    c_genie = conj(h_RU) .* a_BR;
    theta_genie = -angle(c_genie);
    Theta_genie = diag(exp(1i*theta_genie));

    h_BS_genie = h_RU' * Theta_genie * H_BR;
    w_genie    = h_BS_genie' / norm(h_BS_genie');
    h_eff_genie = h_BS_genie * w_genie;

    Pr_genie_W = THzEnv.Pt_W * abs(h_eff_genie)^2;
    SNR_genie  = Pr_genie_W / THzEnv.Noise_W;
    Rate_genie = THzEnv.B * log2(1 + SNR_genie);

    % ---- Reward: ratio to genie-optimal, O(1) per step ----
    % Previously divided by StepsPerEpisode to keep the CUMULATIVE episode
    % reward in [0,1] - but that shrank the per-step reward to ~0.01-0.02,
    % which made EntropyLossWeight (tuned for an O(1) reward) dominate the
    % loss. That's what caused the std to blow up to ~120 and the mean to
    % saturate at a constant: maximizing entropy became cheaper than
    % solving the actual control problem. Keeping reward at O(1) per step
    % and adjusting the stop-criterion threshold instead (see trainOpts)
    % fixes the imbalance at the source instead of just capping symptoms.
    Reward = DataRate / max(Rate_genie, eps);

    LoggedSignals.steps = LoggedSignals.steps + 1;
    LoggedSignals.theta_RU = min(80, max(10, theta_RU + randn() * 2));

    Observation = [sind(LoggedSignals.theta_RU); cosd(LoggedSignals.theta_RU)];
    IsDone = LoggedSignals.steps >= 50;
end