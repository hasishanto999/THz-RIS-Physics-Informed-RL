clear;
clc;
close all;

%% REPOSITORY PATHS
repoRoot = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(repoRoot, 'results');
agentsDir  = fullfile(repoRoot, 'trained_agents');


%% =========================================================
% STAGE 2: PATH LOSS + MOLECULAR ABSORPTION
%
% Computes free-space path loss and molecular absorption at 300 GHz
% using the actual ITU-R P.676-12 Annex 1 line-by-line model (oxygen +
% water vapour resonance lines + dry/wet continuum), not a placeholder
% constant. This is the same formula/coefficients verified earlier
% against the ITU document directly (see conversation), translated to
% MATLAB and packaged to match the Stage 6-9 script format.
%
% Standard atmosphere: p_total = 1013.25 hPa, T = 15 C = 288.15 K,
% water vapour density = 7.5 g/m^3.
% =========================================================

fc = 300;        % GHz
T  = 288.15;      % K
rho = 7.5;        % g/m^3 (water vapour density)
p_total = 1013.25; % hPa

theta = 300.0 / T;
e = rho * T / 216.7;          % water vapour partial pressure, ITU eq (4)
p = p_total - e;              % dry air pressure

% ---- Table 1: oxygen lines (f0, a1, a2, a3, a4, a5, a6) ----
oxygen = [
50.474214,0.975,9.651,6.690,0.0,2.566,6.850;
50.987745,2.529,8.653,7.170,0.0,2.246,6.800;
51.503360,6.193,7.709,7.640,0.0,1.947,6.729;
52.021429,14.320,6.819,8.110,0.0,1.667,6.640;
52.542418,31.240,5.983,8.580,0.0,1.388,6.526;
53.066934,64.290,5.201,9.060,0.0,1.349,6.206;
53.595775,124.600,4.474,9.550,0.0,2.227,5.085;
54.130025,227.300,3.800,9.960,0.0,3.170,3.750;
54.671180,389.700,3.182,10.370,0.0,3.558,2.654;
55.221384,627.100,2.618,10.890,0.0,2.560,2.952;
55.783815,945.300,2.109,11.340,0.0,-1.172,6.135;
56.264774,543.400,0.014,17.030,0.0,3.525,-0.978;
56.363399,1331.800,1.654,11.890,0.0,-2.378,6.547;
56.968211,1746.600,1.255,12.230,0.0,-3.545,6.451;
57.612486,2120.100,0.910,12.620,0.0,-5.416,6.056;
58.323877,2363.700,0.621,12.950,0.0,-1.932,0.436;
58.446588,1442.100,0.083,14.910,0.0,6.768,-1.273;
59.164204,2379.900,0.387,13.530,0.0,-6.561,2.309;
59.590983,2090.700,0.207,14.080,0.0,6.957,-0.776;
60.306056,2103.400,0.207,14.150,0.0,-6.395,0.699;
60.434778,2438.000,0.386,13.390,0.0,6.342,-2.825;
61.150562,2479.500,0.621,12.920,0.0,1.014,-0.584;
61.800158,2275.900,0.910,12.630,0.0,5.014,-6.619;
62.411220,1915.400,1.255,12.170,0.0,3.029,-6.759;
62.486253,1503.000,0.083,15.130,0.0,-4.499,0.844;
62.997984,1490.200,1.654,11.740,0.0,1.856,-6.675;
63.568526,1078.000,2.108,11.340,0.0,0.658,-6.139;
64.127775,728.700,2.617,10.880,0.0,-3.036,-2.895;
64.678910,461.300,3.181,10.380,0.0,-3.968,-2.590;
65.224078,274.000,3.800,9.960,0.0,-3.528,-3.680;
65.764779,153.000,4.473,9.550,0.0,-2.548,-5.002;
66.302096,80.400,5.200,9.060,0.0,-1.660,-6.091;
66.836834,39.800,5.982,8.580,0.0,-1.680,-6.393;
67.369601,18.560,6.818,8.110,0.0,-1.956,-6.475;
67.900868,8.172,7.708,7.640,0.0,-2.216,-6.545;
68.431006,3.397,8.652,7.170,0.0,-2.492,-6.600;
68.960312,1.334,9.650,6.690,0.0,-2.773,-6.650;
118.750334,940.300,0.010,16.640,0.0,-0.439,0.079;
368.498246,67.400,0.048,16.400,0.0,0.000,0.000;
424.763020,637.700,0.044,16.400,0.0,0.000,0.000;
487.249273,237.400,0.049,16.000,0.0,0.000,0.000;
715.392902,98.100,0.145,16.000,0.0,0.000,0.000;
773.839490,572.300,0.141,16.200,0.0,0.000,0.000;
834.145546,183.100,0.145,14.700,0.0,0.000,0.000];

% ---- Table 2: water vapour lines (f0, b1, b2, b3, b4, b5, b6) ----
water = [
22.235080,.1079,2.144,26.38,.76,5.087,1.00;
67.803960,.0011,8.732,28.58,.69,4.930,.82;
119.995940,.0007,8.353,29.48,.70,4.780,.79;
183.310087,2.273,.668,29.06,.77,5.022,.85;
321.225630,.0470,6.179,24.04,.67,4.398,.54;
325.152888,1.514,1.541,28.23,.64,4.893,.74;
336.227764,.0010,9.825,26.93,.69,4.740,.61;
380.197353,11.67,1.048,28.11,.54,5.063,.89;
390.134508,.0045,7.347,21.52,.63,4.810,.55;
437.346667,.0632,5.048,18.45,.60,4.230,.48;
439.150807,.9098,3.595,20.07,.63,4.483,.52;
443.018343,.1920,5.048,15.55,.60,5.083,.50;
448.001085,10.41,1.405,25.64,.66,5.028,.67;
470.888999,.3254,3.597,21.34,.66,4.506,.65;
474.689092,1.260,2.379,23.20,.65,4.804,.64;
488.490108,.2529,2.852,25.86,.69,5.201,.72;
503.568532,.0372,6.731,16.12,.61,3.980,.43;
504.482692,.0124,6.731,16.12,.61,4.010,.45;
547.676440,.9785,.158,26.00,.70,4.500,1.00;
552.020960,.1840,.158,26.00,.70,4.500,1.00;
556.935985,497.0,.159,30.86,.69,4.552,1.00;
620.700807,5.015,2.391,24.38,.71,4.856,.68;
645.766085,.0067,8.633,18.00,.60,4.000,.50;
658.005280,.2732,7.816,32.10,.69,4.140,1.00;
752.033113,243.4,.396,30.86,.68,4.352,.84;
841.051732,.0134,8.177,15.90,.33,5.760,.45;
859.965698,.1325,8.055,30.60,.68,4.090,.84;
899.303175,.0547,7.914,29.85,.68,4.530,.90;
902.611085,.0386,8.429,28.65,.70,5.100,.95;
906.205957,.1836,5.110,24.08,.70,4.700,.53;
916.171582,8.400,1.441,26.73,.70,5.150,.78;
923.112692,.0079,10.293,29.00,.70,5.000,.80;
970.315022,9.009,1.919,25.50,.64,4.940,.67;
987.926764,134.6,.257,29.85,.68,4.550,.90;
1780.000000,17506.,.952,196.3,2.00,24.15,5.00];

lineShape = @(f,f0,delta_f,delta) (f/f0) * ...
    ( (delta_f - delta*(f0-f))/((f0-f)^2+delta_f^2) + ...
      (delta_f - delta*(f0+f))/((f0+f)^2+delta_f^2) );

N_O2 = 0;
for i = 1:size(oxygen,1)
    f0=oxygen(i,1); a1=oxygen(i,2); a2=oxygen(i,3); a3=oxygen(i,4);
    a4=oxygen(i,5); a5=oxygen(i,6); a6=oxygen(i,7);
    S = a1*1e-7 * p * theta^3 * exp(a2*(1-theta));
    delta_f = a3*1e-4 * (p*theta^(0.8-a4) + 1.1*e*theta);
    delta_f = sqrt(delta_f^2 + 2.25e-6);
    delta = (a5 + a6*theta)*1e-4*(p+e)*theta^0.8;
    N_O2 = N_O2 + S*lineShape(fc,f0,delta_f,delta);
end

N_H2O = 0;
for i = 1:size(water,1)
    f0=water(i,1); b1=water(i,2); b2=water(i,3); b3=water(i,4);
    b4=water(i,5); b5=water(i,6); b6=water(i,7);
    S = b1*1e-1 * e * theta^3.5 * exp(b2*(1-theta));
    delta_f = b3*1e-4*(p*theta^b4 + b5*e*theta^b6);
    delta_f = 0.535*delta_f + sqrt(0.217*delta_f^2 + 2.1316e-12*f0^2/theta);
    delta = 0;
    N_H2O = N_H2O + S*lineShape(fc,f0,delta_f,delta);
end

d_cont = 5.6e-4*(p+e)*theta^0.8;
N_D = fc*p*theta^2 * ( 6.14e-5/(d_cont*(1+(fc/d_cont)^2)) + ...
      1.4e-12*p*theta^1.5/(1+1.9e-5*fc^1.5) );

N_total = N_O2 + N_D + N_H2O;
gamma_dB_per_km = 0.1820 * fc * N_total;   % ITU-R P.676-12 eq (1)
gamma_dB_per_m  = gamma_dB_per_km / 1000;
k_abs_Np_per_m  = gamma_dB_per_m / 4.343;   % Nepers/m, for use in 4.343*k_abs*d formula

%% ---- Free-space path loss + total path loss vs distance ----

c = 3e8;
lambda = c / (fc*1e9);

d_link = 0.15;   % matches Stage 3-9 system parameters
FSPL_dB = 20*log10(4*pi*d_link/lambda);
AbsLoss_dB = gamma_dB_per_m * d_link;
TotalPL_dB = FSPL_dB + AbsLoss_dB;

fprintf('\n============================================\n');
fprintf(' STAGE 2: PATH LOSS + MOLECULAR ABSORPTION\n');
fprintf('============================================\n\n');
fprintf('Carrier Frequency = %.0f GHz\n', fc);
fprintf('Standard atmosphere: p_total=%.2f hPa, T=%.2f K, rho=%.1f g/m^3\n', p_total, T, rho);
fprintf('theta = %.4f, e = %.4f hPa, p(dry) = %.4f hPa\n\n', theta, e, p);

fprintf('--- ITU-R P.676-12 line-by-line result ---\n');
fprintf('Specific attenuation gamma = %.4f dB/km  (%.6f dB/m)\n', gamma_dB_per_km, gamma_dB_per_m);
fprintf('Equivalent k_abs           = %.6f Np/m  (for use in 4.343*k_abs*d formula)\n\n', k_abs_Np_per_m);

fprintf('--- Path loss at d = %.2f m (matches Stage 3-9 link distance) ---\n', d_link);
fprintf('Free-space path loss (FSPL) = %.4f dB\n', FSPL_dB);
fprintf('Molecular absorption loss   = %.6f dB\n', AbsLoss_dB);
fprintf('Total path loss             = %.4f dB\n', TotalPL_dB);
fprintf('Absorption as %% of total    = %.4f%%\n\n', 100*AbsLoss_dB/TotalPL_dB);

fprintf('============================================\n');
fprintf(' STAGE 2 COMPLETE\n');
fprintf('============================================\n');

%% ---- Save for use by later stages ----
save(fullfile(resultsDir, 'stage2_results.mat'), 'k_abs_Np_per_m', 'gamma_dB_per_km', 'fc', 'T', 'rho', 'p_total');
fprintf('Saved to stage2_results.mat (k_abs_Np_per_m available for Stages 6-9 if desired)\n');

%% ---- Plot: absorption contribution vs distance ----
d_sweep = 0.01:0.01:2;
FSPL_sweep = 20*log10(4*pi*d_sweep/lambda);
Abs_sweep = gamma_dB_per_m * d_sweep;

figure;
subplot(1,2,1);
plot(d_sweep, FSPL_sweep, 'b-', 'LineWidth', 1.5); hold on;
plot(d_sweep, FSPL_sweep + Abs_sweep, 'r--', 'LineWidth', 1.5);
xlabel('Distance (m)'); ylabel('Path Loss (dB)');
legend('Free-space only', 'Free-space + absorption', 'Location', 'best');
title('Path loss vs. distance at 300 GHz');
grid on;

subplot(1,2,2);
plot(d_sweep, Abs_sweep, 'k-', 'LineWidth', 1.5);
xlabel('Distance (m)'); ylabel('Absorption loss (dB)');
title(sprintf('Molecular absorption contribution (gamma=%.3f dB/km)', gamma_dB_per_km));
grid on;
