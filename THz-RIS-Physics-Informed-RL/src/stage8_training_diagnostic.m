%% REPOSITORY PATHS
repoRoot = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(repoRoot, 'results');
agentsDir  = fullfile(repoRoot, 'trained_agents');

%% STAGE 8 DIAGNOSTIC: true mean/std of the trained policy (no sampling noise)

load(fullfile(agentsDir, 'stage8_trained_agent.mat'), 'agent', 'THzEnv');   % works even after clear all

theta_sweep = 10:5:80;
learned_slope = zeros(size(theta_sweep));
true_slope    = zeros(size(theta_sweep));
raw_mean      = zeros(size(theta_sweep));
raw_std       = zeros(size(theta_sweep));

% getAction ALWAYS samples from the Gaussian, regardless of
% UseExplorationPolicy. evaluate() on the actor object returns the true
% mean/std with zero sampling noise - this is the documented way to do it.
actor_obj = getActor(agent);

for i = 1:numel(theta_sweep)
    theta_RU = theta_sweep(i);
    obs = {[sind(theta_RU); cosd(theta_RU)]};

    out = evaluate(actor_obj, obs);   % {mean, std}
    raw_mean(i) = out{1};
    raw_std(i)  = out{2};

    learned_slope(i) = raw_mean(i) * pi;
    true_slope(i) = pi * (sind(theta_RU) - sind(THzEnv.theta_BR));
end

fprintf('\n=== TRUE MEAN/STD CHECK (no sampling) ===\n');
fprintf('min/max raw_mean = %.4f / %.4f   (should be within [-1,1])\n', min(raw_mean), max(raw_mean));
fprintf('min/max raw_std  = %.4f / %.4f\n\n', min(raw_std), max(raw_std));

figure;
subplot(2,1,1);
plot(theta_sweep, true_slope, 'k-o', 'DisplayName', 'True optimal slope');
hold on;
plot(theta_sweep, learned_slope, 'r--x', 'DisplayName', 'Learned MEAN slope (no noise)');
xlabel('\theta_{RU} (deg)'); ylabel('RIS phase-gradient slope (rad)');
legend('Location', 'best');
title('True mean policy vs. optimal slope'); grid on;

subplot(2,1,2);
plot(theta_sweep, raw_std, 'm-s');
xlabel('\theta_{RU} (deg)'); ylabel('Learned std (raw)');
title('Actor standard deviation across angles'); grid on;

fprintf('=== DIAGNOSTIC SUMMARY ===\n');
fprintf('Correlation(learned mean, true) = %.4f\n', corr(learned_slope(:), true_slope(:)));
fprintf('Std of learned mean across angles = %.4f\n', std(learned_slope));
fprintf('Std of true_slope across angles   = %.4f\n', std(true_slope));
