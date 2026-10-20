function varargout = thz_lib(fcn, varargin)
%THZ_LIB  Shared physics engine + helpers for the 300 GHz RIS / PPO project.
%
% WHY THIS FILE EXISTS
%   Training (stage8_multiseed_train.m) and evaluation (stage9_*.m) must use
%   EXACTLY the same channel physics. Previously the channel model was copy-
%   pasted into 6 scripts; adding near-field + broken elements to all of them
%   by hand would invite mismatches. Now there is ONE copy, here.
%
% HOW TO CALL (all commands go through the first argument)
%   cfg = thz_lib('defaultcfg')
%   n   = thz_lib('actdim', cfg)                  % actor output size
%   ph  = thz_lib('action2phases', cfg, action, theta_est)  % RL action -> 16x1 RIS phases [rad]
%                                                 % (theta_est [deg] only used by '16DR')
%   ph  = thz_lib('eq4phases', cfg, theta_est)    % closed-form (Eq. 4) far-field phases
%   ph  = thz_lib('nfphases', cfg, theta_est)     % analytic NEAR-FIELD conjugate phases (knows d_RU, d_BR; ignores faults)
%   ph  = thz_lib('nffaultphases', cfg, theta_est)% analytic near-field + stuck-element-aware phases (strong baseline)
%   [rate_gbps, info] = thz_lib('evaluate', cfg, theta_true, phases)
%   r   = thz_lib('agentrates', cfg, actor, theta_true, theta_est)
%   [m, halfwidth] = thz_lib('ci95', x)           % mean and 95% CI half-width
%   thz_lib('selftest')                           % RUN THIS FIRST (no toolbox needed)
%
% PHYSICS MODEL (what 'evaluate' does)
%   * RIS = uniform linear array on a line, element n at x_n = n*d_RIS.
%   * nearfield = false : plane-wave (far-field) steering vectors, identical to
%                         the original Stage 7/8 code  (a_n = exp(+j k0 x_n sin(theta)))
%   * nearfield = true  : spherical wave. Each element gets its own Euclidean
%                         distance d_n to the source/user, phase = -k0*(d_n - r).
%                         (Reference-subtracted, so it reduces EXACTLY to the
%                         plane-wave vector when r -> infinity.)
%                         Applied to BOTH RIS-side links (BS->RIS and RIS->User).
%                         Amplitude stays uniform (path loss uses the centre distance).
%   * broken_idx        : these RIS elements are STUCK at phase 0 (they still
%                         reflect, but ignore the commanded phase). Applied
%                         inside 'evaluate', so every method is hit identically.
%   * BS array (8 el.)  : kept far-field (aperture only 3.5 mm) with MRT beamforming.
%   * Reward normaliser : rate_genie_ach = best rate ACHIEVABLE with the stuck
%                         elements (working elements align to the phase of the
%                         stuck elements' resultant). rate_genie_free = ideal
%                         rate if no element were broken (not reachable).
%
% NOTE: elements are numbered 1..16 in MATLAB, so broken_idx = [3 12] means the
%       3rd and 12th elements.

switch lower(fcn)
    case 'defaultcfg'
        varargout{1} = default_cfg();
    case 'actdim'
        varargout{1} = action_dim(varargin{:});
    case 'action2phases'
        varargout{1} = action_to_phases(varargin{:});
    case 'eq4phases'
        varargout{1} = eq4_phases(varargin{:});
    case 'nfphases'
        varargout{1} = nf_phases(varargin{:});
    case 'nffaultphases'
        varargout{1} = nf_fault_phases(varargin{:});
    case 'evaluate'
        [varargout{1:max(nargout,1)}] = evaluate_link(varargin{:});
    case 'agentrates'
        varargout{1} = agent_rates(varargin{:});
    case 'ci95'
        [varargout{1:max(nargout,1)}] = ci95(varargin{:});
    case 'selftest'
        selftest();
    otherwise
        error('thz_lib:unknownCommand', 'Unknown command "%s".', fcn);
end
end


%% =====================================================================
function cfg = default_cfg()
% All constants identical to the original Stage 8 script, plus the new
% "realism" switches (nearfield, broken_idx, sigma_obs_deg).
    cfg.fc      = 300e9;
    cfg.lambda  = 3e8 / cfg.fc;
    cfg.B       = 10e9;
    cfg.Pt_W    = 10^((20-30)/10);                 % 20 dBm
    cfg.Noise_W = 1.380649e-23 * 290 * cfg.B;
    cfg.k_abs   = 1.198e-3;                        % ITU-R P.676-12 value (Stage 2)

    cfg.N_BS    = 8;
    cfg.N_RIS   = 16;
    cfg.d_BS    = cfg.lambda/2;
    cfg.d_RIS   = cfg.lambda/2;

    cfg.d_BR    = 0.15;    % BS -> RIS distance [m]
    cfg.d_RU    = 0.05;    % RIS -> user distance [m]  (0.05 m is well inside the
                           % RIS Rayleigh distance ~0.11 m, so curvature matters)
    cfg.theta_BS = 20;     % [deg]
    cfg.theta_BR = 20;     % [deg]

    cfg.theta_min     = 10;   % user-angle range [deg]
    cfg.theta_max     = 80;
    cfg.walk_std_deg  = 2;    % random-walk step of the user angle per RL step
    cfg.StepsPerEpisode = 50;

    % ---- Realism switches -------------------------------------------
    cfg.nearfield      = true;      % spherical-wave RIS links
    cfg.broken_idx     = [3 12];    % stuck-at-0 RIS elements
    cfg.sigma_obs_deg  = 5;         % std of the angle the agent OBSERVES [deg]
    cfg.phase_offset   = [];        % OPTIONAL unmodelled hardware phase error per element [rad], N_RIS x 1.
                                    % Added to the commanded phases inside 'evaluate'. The analytic
                                    % controllers do not know it; a learned 16-D controller can.
                                    % [] = off (reproduces the paper's environment exactly).

    % ---- Agent type ---------------------------------------------------
    cfg.agent_type     = '16D';     % '1D' | '2D' | '16D' | '16DR' | '1DR' | '2DR'  (xR = residual around Eq.4) | '16DS' (structured: slope+curvature+per-element)
    cfg.lowdim_cossin  = false;     % if true, 1D/2D params are also sent as [cos,sin]->atan2
    cfg.res_scale      = pi;        % '16DR': action in [-1,1] -> correction in [-pi,pi] rad around Eq.4
    cfg.res_slope      = pi/4;      % '1DR'/'2DR': slope correction range [rad per element]
    cfg.res_curv       = 2*pi;      % '2DR'/'16DS': curvature correction range [rad at the array edge]
    cfg.res_elem       = pi;        % '16DS': per-element correction range [rad]
end


%% =====================================================================
function n = action_dim(cfg)
% Number of actor OUTPUTS (network action-vector length).
    switch cfg.agent_type
        case '1D'
            n = 1 + double(cfg.lowdim_cossin);          % slope
        case '2D'
            n = 2 * (1 + double(cfg.lowdim_cossin));    % slope + curvature
        case '16D'
            n = 2 * cfg.N_RIS;                          % [cos x16 ; sin x16]
        case '16DR'
            n = cfg.N_RIS;                              % one correction per element
        case '16DS'
            n = 2 + cfg.N_RIS;                          % slope + curvature + one correction per element
        case '1DR'
            n = 1;                                      % slope correction around Eq.4
        case '2DR'
            n = 2;                                      % slope + curvature correction around Eq.4
        otherwise
            error('thz_lib:agentType', 'Unknown agent_type "%s".', cfg.agent_type);
    end
end


%% =====================================================================
function phases = action_to_phases(cfg, a, theta_est)
% Convert the raw actor action (each entry in [-1,1]) to N_RIS phases [rad].
%
%  1D : phase_n = n * slope                         slope = pi*a1
%  2D : phase_n = n * slope + curv * (n/(N-1))^2    slope = pi*a1, curv = 2*pi*a2
%       (a quadratic phase profile is exactly what a spherical wavefront needs)
%  16D: phase_n = atan2( sin_n , cos_n )            action = [cos_1..cos_N, sin_1..sin_N]
%       <-- THE PHASE-WRAP FIX: the network never outputs an angle directly, so
%           there is no discontinuity at +/-pi for the optimiser to fight.
%  1DR : phase_n = Eq4_phase_n(theta_est) + n * res_slope * a1                       (RESIDUAL 1-D)
%  2DR : phase_n = Eq4_phase_n(theta_est) + n * res_slope * a1 + res_curv * a2 * (n/(N-1))^2   (RESIDUAL 2-D)
%  16DS: phase_n = Eq4_phase_n(theta_est) + n*res_slope*a1 + res_curv*a2*(n/(N-1))^2 + res_elem*a_(2+n)
%        (STRUCTURED 16-D: the 2DR corrections + a per-element correction. Starts at Eq.4.)
%  16DR: phase_n = Eq4_phase_n(theta_est) + res_scale * a_n      (RESIDUAL 16-D)
%       The agent learns a per-element CORRECTION to the closed-form solution.
%       With a = 0 it reproduces Eq. 4 exactly, so training starts at the
%       baseline instead of at random phases. Needs theta_est (the observed angle).
    a = max(-1, min(1, a(:)));
    N = cfg.N_RIS;
    n = (0:N-1).';
    if nargin < 3, theta_est = []; end
    switch cfg.agent_type
        case '1D'
            if cfg.lowdim_cossin
                slope = atan2(a(2), a(1));
            else
                slope = pi * a(1);
            end
            phases = n * slope;
        case '2D'
            if cfg.lowdim_cossin
                slope = atan2(a(2), a(1));
                curv  = 2 * atan2(a(4), a(3));
            else
                slope = pi * a(1);
                curv  = 2*pi * a(2);
            end
            phases = n * slope + curv * (n/(N-1)).^2;
        case '16D'
            phases = atan2(a(N+1:2*N), a(1:N));
        case '16DR'
            if isempty(theta_est)
                error('thz_lib:needAngle', '16DR needs theta_est (observed angle) as 3rd argument.');
            end
            if isfield(cfg, 'res_scale'), rs = cfg.res_scale; else, rs = pi; end
            phases = eq4_phases(cfg, theta_est) + rs * a(1:N);
        case {'1DR', '2DR'}
            if isempty(theta_est)
                error('thz_lib:needAngle', '%s needs theta_est (observed angle) as 3rd argument.', cfg.agent_type);
            end
            if isfield(cfg, 'res_slope'), rs1 = cfg.res_slope; else, rs1 = pi/4; end
            if isfield(cfg, 'res_curv'),  rc  = cfg.res_curv;  else, rc  = 2*pi; end
            phases = eq4_phases(cfg, theta_est) + n * (rs1 * a(1));
            if strcmp(cfg.agent_type, '2DR')
                phases = phases + rc * a(2) * (n/(N-1)).^2;
            end
        case '16DS'
            if isempty(theta_est)
                error('thz_lib:needAngle', '16DS needs theta_est (observed angle) as 3rd argument.');
            end
            if isfield(cfg, 'res_slope'), rs1 = cfg.res_slope; else, rs1 = pi/4; end
            if isfield(cfg, 'res_curv'),  rc  = cfg.res_curv;  else, rc  = 2*pi; end
            if isfield(cfg, 'res_elem'),  re  = cfg.res_elem;  else, re  = pi;   end
            phases = eq4_phases(cfg, theta_est) + n * (rs1 * a(1)) ...
                     + rc * a(2) * (n/(N-1)).^2 + re * a(3:2+N);
        otherwise
            error('thz_lib:agentType', 'Unknown agent_type "%s".', cfg.agent_type);
    end
end


%% =====================================================================
function phases = eq4_phases(cfg, theta_est_deg)
% Closed-form baseline (paper Eq. 4 / Eq. 23-24): PLANE-WAVE, FAR-FIELD phase
% law computed from a (possibly noisy) angle ESTIMATE. It knows nothing about
% near-field curvature or broken elements.
    n    = (0:cfg.N_RIS-1).';
    k0d  = 2*pi*cfg.d_RIS/cfg.lambda;
    a_RU = exp(1i * k0d * n * sind(theta_est_deg));
    a_BR = exp(1i * k0d * n * sind(cfg.theta_BR));
    phases = -angle(conj(a_RU) .* a_BR);
end


%% =====================================================================
function phases = nf_phases(cfg, theta_est_deg)
% Analytic NEAR-FIELD conjugate phase from the ESTIMATED angle. It uses the
% spherical-wave model with the known distances d_BR and d_RU, but ignores the
% stuck elements. Same information the learned controllers absorb in training.
    [a_BR, a_RU] = ris_vectors(cfg, theta_est_deg);
    phases = -angle(conj(a_RU) .* a_BR);
end

function phases = nf_fault_phases(cfg, theta_est_deg)
% Analytic near-field + stuck-element-aware phase (Eq. 6 logic): every working
% element aligns with the fixed resultant S of the stuck elements. With the TRUE
% angle this reaches rate_genie_ach exactly; with a noisy angle it is the strong
% analytic baseline the learned residual controllers must beat.
    [a_BR, a_RU] = ris_vectors(cfg, theta_est_deg);
    c = conj(a_RU) .* a_BR;
    b = unique(cfg.broken_idx(:));
    if isempty(b), S = 0; else, S = sum(c(b)); end
    phases = angle(S) - angle(c);
end


%% =====================================================================
function [a_BR, a_RU] = ris_vectors(cfg, theta_RU_true)
% RIS-side response vectors for the BS->RIS and RIS->user links.
    n  = (0:cfg.N_RIS-1).';
    k0 = 2*pi/cfg.lambda;
    x  = n * cfg.d_RIS;                 % element positions along the array
    if cfg.nearfield
        a_BR = spherical_vec(x, cfg.d_BR, cfg.theta_BR,   k0);
        a_RU = spherical_vec(x, cfg.d_RU, theta_RU_true,  k0);
    else
        a_BR = exp(1i * k0 * x * sind(cfg.theta_BR));
        a_RU = exp(1i * k0 * x * sind(theta_RU_true));
    end
end

function a = spherical_vec(x, r, theta_deg, k0)
% Source at distance r, angle theta from the array broadside, measured from
% element 0 (x = 0). Exact Euclidean distance to every element.
    px = r * sind(theta_deg);
    pz = r * cosd(theta_deg);
    d  = sqrt((px - x).^2 + pz^2);
    a  = exp(-1i * k0 * (d - r));       % common phase exp(-j k0 r) dropped
end

function G = path_gain(d, cfg)
    PL_dB = 20*log10(4*pi*d/cfg.lambda) + 4.343*cfg.k_abs*d;
    G = 10^(-PL_dB/10);
end


%% =====================================================================
function [rate_gbps, info] = evaluate_link(cfg, theta_RU_true, phases)
% Data rate [Gbps] of the BS-RIS-user link for the given RIS phases, the TRUE
% user angle, near/far-field per cfg, and stuck elements per cfg.broken_idx.
    N = cfg.N_RIS;
    phases = phases(:);
    if isfield(cfg, 'phase_offset') && ~isempty(cfg.phase_offset)
        phases = phases + cfg.phase_offset(:);      % unmodelled hardware phase error (applied BEFORE the stuck override)
    end
    b = unique(cfg.broken_idx(:));
    if ~isempty(b)
        if any(b < 1 | b > N)
            error('thz_lib:brokenIdx', 'broken_idx must lie in 1..%d', N);
        end
        phases(b) = 0;                              % stuck-at-0 hardware fault
    end

    [a_BR, a_RU] = ris_vectors(cfg, theta_RU_true);
    g = sqrt(path_gain(cfg.d_BR, cfg) * path_gain(cfg.d_RU, cfg)) * sqrt(cfg.N_BS);

    % With H_BR = sqrt(G_BR) a_BR a_BS^H, h_RU = sqrt(G_RU) a_RU and MRT at the
    % BS (unit-modulus a_BS), the effective gain reduces to:
    c     = conj(a_RU) .* a_BR;                     % per-element channel phasor
    h_eff = g * abs(sum(c .* exp(1i*phases)));
    rate_gbps = to_rate(cfg, h_eff);

    % ---- Genie rates -------------------------------------------------
    S      = sum(c(b));                             % stuck elements' fixed resultant (0 if none)
    K      = N - numel(b);                          % working elements
    h_ach  = g * (abs(S) + K);                      % best achievable with stuck elements
    h_free = g * N;                                 % ideal, no broken elements

    info.rate_genie_ach  = to_rate(cfg, h_ach);
    info.rate_genie_free = to_rate(cfg, h_free);
    info.snr_lin         = cfg.Pt_W * h_eff^2 / cfg.Noise_W;
end

function r = to_rate(cfg, h)
    r = cfg.B * log2(1 + cfg.Pt_W * h^2 / cfg.Noise_W) / 1e9;
end


%% =====================================================================
function r = agent_rates(cfg, actor, theta_true, theta_est)
% Deterministic (mean-action) rate of a trained actor. The actor only sees
% theta_est; the physical channel uses theta_true.
    persistent warned
    if isempty(warned), warned = false; end
    M = numel(theta_true);
    r = zeros(M, 1);
    theta_est = theta_est(:);

    % ---- FAST PATH: one batched forward pass (obs = 2 x M) ------------------
    % Verified against the single-sample path on 3 samples. Any failure falls back to
    % the per-sample loop below (identical results, just slower).
    try
        out = evaluate(actor, {[sind(theta_est).'; cosd(theta_est).']});
        A   = reshape(out{1}, [], M);                          % nAct x M mean actions
        for i = 1:min(3, M)
            o1 = evaluate(actor, {[sind(theta_est(i)); cosd(theta_est(i))]});
            if max(abs(o1{1}(:) - A(:, i))) > 1e-4
                error('thz_lib:batchMismatch', 'batched and single-sample actions differ');
            end
        end
        for i = 1:M
            r(i) = evaluate_link(cfg, theta_true(i), action_to_phases(cfg, A(:, i), theta_est(i)));
        end
        return;
    catch ME
        if ~warned
            warning('thz_lib:slowPath', 'Batched actor evaluation unavailable (%s); using the per-sample loop.', ME.message);
            warned = true;
        end
    end

    % ---- SLOW PATH (original behaviour) --------------------------------------
    for i = 1:M
        obs = {[sind(theta_est(i)); cosd(theta_est(i))]};
        out = evaluate(actor, obs);        % {mean, std} -> out{1} = mean action
        act = out{1};
        ph  = action_to_phases(cfg, act(:), theta_est(i));
        r(i) = evaluate_link(cfg, theta_true(i), ph);
    end
end


%% =====================================================================
function [m, hw] = ci95(x)
% Mean and 95% confidence-interval half-width (Student-t).
    x = x(:);
    n = numel(x);
    m = mean(x);
    if n < 2
        hw = NaN;
        return;
    end
    if exist('tinv', 'file')
        t = tinv(0.975, n-1);
    else
        t = 1.96;
        warning('thz_lib:noStats', 'tinv not found (Statistics Toolbox); using z = 1.96.');
    end
    hw = t * std(x) / sqrt(n);
end


%% =====================================================================
function selftest()
    fprintf('\n=== thz_lib self-test ===\n');
    rng(1);
    allok = true;
    cfg = default_cfg();

    % 1. Near-field -> far-field limit
    cN = cfg;  cN.nearfield = true;  cN.d_BR = 1e3;  cN.d_RU = 1e3;
    cF = cfg;  cF.nearfield = false;
    [bn, un] = ris_vectors(cN, 40);
    [bf, uf] = ris_vectors(cF, 40);
    e = max([max(abs(bn-bf)), max(abs(un-uf))]);
    allok = report('Spherical wave -> plane wave as r -> inf', e < 1e-2, sprintf('max err = %.2e', e)) && allok;

    % 2. Eq. 4 with the true angle is optimal in the clean far-field world
    cC = cfg;  cC.nearfield = false;  cC.broken_idx = [];
    [r, inf_] = evaluate_link(cC, 40, eq4_phases(cC, 40));
    allok = report('Eq.4 = genie in ideal environment', abs(r - inf_.rate_genie_free) < 1e-9*max(1,r), ...
                   sprintf('%.6f vs %.6f Gbps', r, inf_.rate_genie_free)) && allok;

    % 3. Broken elements ignore their commanded phase
    p  = 2*pi*rand(cfg.N_RIS, 1);
    p2 = p;  p2(cfg.broken_idx) = 0;
    r1 = evaluate_link(cfg, 40, p);
    r2 = evaluate_link(cfg, 40, p2);
    allok = report('Broken elements are stuck at 0 phase', abs(r1 - r2) < 1e-12*max(1,r1), '') && allok;

    % 4. Achievable genie is really the maximum (and is reachable)
    [b40, u40] = ris_vectors(cfg, 40);
    c   = conj(u40) .* b40;
    S   = sum(c(cfg.broken_idx));
    opt = angle(S) - angle(c);
    [r_opt, inf40] = evaluate_link(cfg, 40, opt);
    okA = abs(r_opt - inf40.rate_genie_ach) < 1e-9*max(1,r_opt);
    mx  = 0;
    for k = 1:300
        mx = max(mx, evaluate_link(cfg, 40, 2*pi*rand(cfg.N_RIS,1)));
    end
    okB = mx <= inf40.rate_genie_ach + 1e-9;
    okC = inf40.rate_genie_ach <= inf40.rate_genie_free + 1e-9;
    allok = report('Achievable genie reachable', okA, sprintf('%.4f Gbps', r_opt)) && allok;
    allok = report('No random phase beats genie', okB, sprintf('best random %.4f <= %.4f', mx, inf40.rate_genie_ach)) && allok;
    allok = report('Achievable <= ideal (no-fault) genie', okC, '') && allok;

    % 5. Action decoding shapes for every agent type
    types = {'1D','2D','16D','16DR','1DR','2DR','16DS'};
    for t = 1:numel(types)
        for cs = [false true]
            cc = cfg;  cc.agent_type = types{t};  cc.lowdim_cossin = cs;
            a  = 2*rand(action_dim(cc), 1) - 1;
            ph = action_to_phases(cc, a, 40);
            ok = isequal(size(ph), [cfg.N_RIS 1]) && all(isfinite(ph));
            allok = report(sprintf('action2phases %s (cossin=%d), dim=%d', types{t}, cs, action_dim(cc)), ok, '') && allok;
        end
    end

    % 6. Residual agent with zero action must reproduce Eq. 4 exactly
    for rt = {'1DR','2DR','16DR','16DS'}
        cR = cfg;  cR.agent_type = rt{1};
        e6 = max(abs(action_to_phases(cR, zeros(action_dim(cR),1), 33) - eq4_phases(cR, 33)));
        allok = report([rt{1} ' with zero action = Eq. 4'], e6 < 1e-12, '') && allok;
    end

    % 7. Analytic near-field + fault-aware controller reaches the achievable genie (true angle)
    [r7, i7] = evaluate_link(cfg, 33, nf_fault_phases(cfg, 33));
    allok = report('NF+fault-aware analytic = achievable genie (true angle)', ...
                   abs(r7 - i7.rate_genie_ach) < 1e-9*max(1,r7), ...
                   sprintf('%.4f vs %.4f Gbps', r7, i7.rate_genie_ach)) && allok;

    % 8. An unmodelled phase offset is cancelled by commanding (target - offset)
    cO = cfg;  cO.phase_offset = deg2rad(15) * randn(cfg.N_RIS, 1);
    [r8, i8] = evaluate_link(cO, 33, nf_fault_phases(cfg, 33) - cO.phase_offset);
    allok = report('Phase offset is invertible by the commanded phases', ...
                   abs(r8 - i8.rate_genie_ach) < 1e-9*max(1,r8), '') && allok;

    if allok
        fprintf('\nALL SELF-TESTS PASSED.\n\n');
    else
        error('thz_lib:selftest', 'One or more self-tests FAILED - do not train until fixed.');
    end
end

function ok = report(name, ok, extra)
    if ok, tag = 'PASS'; else, tag = 'FAIL'; end
    fprintf('  [%s] %s  %s\n', tag, name, extra);
end