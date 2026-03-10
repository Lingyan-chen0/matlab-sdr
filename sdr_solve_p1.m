function [Rx_opt, w_opt, info] = sdr_solve_p1(params)
%SDR_SOLVE_P1 Basic SDR solver for P1 using CVX.
%   Minimizes an epigraph t for CRB via SDP with SINR and power constraints.
%
% Required fields in params:
%   Nt              - number of transmit antennas
%   Pt              - power budget (trace(Rx) <= Pt)
%   h               - channel matrix, size Nt x Nc
%   Rn              - noise/interference covariances, size Nt x Nt x Nc
%   Gamma           - SINR targets, size Nc x 1
%   sigma_c2        - noise power scalar
%   fisher_builder  - function handle F = fisher_builder(Rx, epsilon)
%
% Optional fields:
%   save_path       - file path string to save results as .mat (e.g. 'results/run1.mat')
%   epsilon_samples - size Q x Ns, robust samples of epsilon
%   epsilon_sampling - struct with fields:
%       .mode        - 'sphere_radial' (default)
%       .num_dirs    - number of sphere directions
%       .num_radii   - number of radial levels (includes 0)
%       .epsilon_dim - dimension Q
%       .epsilon_bar - max radius for ||epsilon||_2
%
% Outputs:
%   Rx_opt  - optimal relaxed covariance
%   w_opt   - rank-1 recovery via principal eigenvector
%   info    - solver status and diagnostics
    Nt = params.Nt;
    Pt = params.Pt;
    h = params.h;
    Rn = params.Rn;
    Gamma = params.Gamma;
    sigma_c2 = params.sigma_c2;

    if ~isfield(params, 'fisher_builder')
        error('params.fisher_builder is required');
    end
    fisher_builder = params.fisher_builder;

    if isfield(params, 'epsilon_samples')
        eps_samples = params.epsilon_samples;
        Ns = size(eps_samples, 2);
    else
        eps_samples = [];
        Ns = 0;
    end

    if isfield(params, 'fim_reg')
        fim_reg = params.fim_reg;
    else
        fim_reg = 0;
    end

    if isfield(params, 'fim_slack_weight')
        fim_slack_weight = params.fim_slack_weight;
    else
        fim_slack_weight = 0;
    end

    if Ns == 0 && isfield(params, 'epsilon_sampling')
        eps_cfg = params.epsilon_sampling;
        if ~isfield(eps_cfg, 'mode')
            eps_cfg.mode = 'sphere_radial';
        end
        if ~strcmp(eps_cfg.mode, 'sphere_radial')
            error('Unsupported epsilon_sampling.mode: %s', eps_cfg.mode);
        end
        if ~isfield(eps_cfg, 'epsilon_dim') || ~isfield(eps_cfg, 'epsilon_bar')
            error('epsilon_sampling requires epsilon_dim and epsilon_bar when epsilon_samples is not provided');
        end
        if ~isfield(eps_cfg, 'num_dirs')
            eps_cfg.num_dirs = 32;
        end
        if ~isfield(eps_cfg, 'num_radii')
            eps_cfg.num_radii = 4;
        end
        eps_samples = generate_epsilon_samples_sphere_radial( ...
            eps_cfg.epsilon_dim, eps_cfg.num_dirs, eps_cfg.num_radii, eps_cfg.epsilon_bar);
        Ns = size(eps_samples, 2);
    end

    % Infer Fisher matrix size for SDP epigraph (constant dimension).
    F0 = fisher_builder(eye(Nt), []);
    if isstruct(F0)
        F0_block = [F0.s11, F0.s21'; F0.s21, F0.S22];
        p = size(F0_block, 1);
        base_scale = norm(full(F0_block), 'fro');
    else
        p = size(F0, 1);
        base_scale = norm(full(F0), 'fro');
    end

    if isfield(params, 'fim_scale')
        fim_scale = params.fim_scale;
    else
        fim_scale = max(1, base_scale);
    end

    skip_crb = isfield(params, 'skip_crb') && params.skip_crb;

    cvx_clear
    if isfield(params, 'cvx_precision') && ~isempty(params.cvx_precision)
        cvx_precision(params.cvx_precision);
    end
    if isfield(params, 'cvx_quiet') && ~params.cvx_quiet
        cvx_begin sdp
    else
        cvx_begin sdp quiet
    end
        variable Rx(Nt, Nt) hermitian semidefinite
        variable t
        variable Z(p, p, max(Ns, 1)) hermitian semidefinite
        if fim_slack_weight > 0
            variable fim_slack
            fim_slack >= 0;
        end
        if skip_crb
            minimize(0)
        else
            if fim_slack_weight > 0
                minimize(t + fim_slack_weight * fim_slack)
            else
                minimize(t)
            end
        end
        subject to
            trace(Rx) <= Pt;
            for n = 1:size(h, 2)
                hn = h(:, n);
                Rn_n = Rn(:, :, n);
                lhs = (1 + 1 / Gamma(n)) * real(hn' * Rn_n * hn);
                rhs = real(hn' * Rx * hn) + sigma_c2;
                lhs >= rhs;
            end
            if ~skip_crb
                if Ns == 0
                    F = fisher_builder(Rx, []);
                    if isstruct(F)
                        F_block = [F.s11, F.s21'; F.s21, F.S22];
                        F_block = (F_block + F_block') / 2;
                        if fim_reg > 0
                            F_block = F_block + fim_reg * eye(p);
                        end
                        if fim_slack_weight > 0
                            F_block = F_block + fim_slack * eye(p);
                        end
                        if fim_scale ~= 1
                            F_block = F_block / fim_scale;
                        end
                        [F_block, eye(p); eye(p), Z(:, :, 1)] == hermitian_semidefinite(2 * p);
                    else
                        F = (F + F') / 2;
                        if fim_reg > 0
                            F = F + fim_reg * eye(p);
                        end
                        if fim_slack_weight > 0
                            F = F + fim_slack * eye(p);
                        end
                        if fim_scale ~= 1
                            F = F / fim_scale;
                        end
                        [F, eye(p); eye(p), Z(:, :, 1)] == hermitian_semidefinite(2 * p);
                    end
                    trace(Z(:, :, 1)) <= t;
                else
                    for k = 1:Ns
                        ek = eps_samples(:, k);
                        F = fisher_builder(Rx, ek);
                        if isstruct(F)
                            F_block = [F.s11, F.s21'; F.s21, F.S22];
                            F_block = (F_block + F_block') / 2;
                            if fim_reg > 0
                                F_block = F_block + fim_reg * eye(p);
                            end
                            if fim_slack_weight > 0
                                F_block = F_block + fim_slack * eye(p);
                            end
                            if fim_scale ~= 1
                                F_block = F_block / fim_scale;
                            end
                            [F_block, eye(p); eye(p), Z(:, :, k)] == hermitian_semidefinite(2 * p);
                        else
                            F = (F + F') / 2;
                            if fim_reg > 0
                                F = F + fim_reg * eye(p);
                            end
                            if fim_slack_weight > 0
                                F = F + fim_slack * eye(p);
                            end
                            if fim_scale ~= 1
                                F = F / fim_scale;
                            end
                            [F, eye(p); eye(p), Z(:, :, k)] == hermitian_semidefinite(2 * p);
                        end
                        trace(Z(:, :, k)) <= t;
                    end
                end
            end
            if skip_crb
                t == 0;
            end
    cvx_end

    Rx_opt = Rx;

    % Rank-1 recovery via principal eigenvector.
    [V, D] = eig(full(Rx));
    [dmax, idx] = max(real(diag(D)));
    w_opt = sqrt(max(dmax, 0)) * V(:, idx);

    info.cvx_status = cvx_status;
    info.cvx_optval = cvx_optval;
    info.t = t;
    info.rank = rank(full(Rx), 1e-6);
    info.fim_scale = fim_scale;
    if fim_scale ~= 1
        info.t_unscaled = t / fim_scale;
    end
    if fim_slack_weight > 0
        info.fim_slack = fim_slack;
    end

    if isfield(params, 'save_path') && ~isempty(params.save_path)
        save_dir = fileparts(params.save_path);
        if ~isempty(save_dir) && ~exist(save_dir, 'dir')
            mkdir(save_dir);
        end
        save(params.save_path, 'Rx_opt', 'w_opt', 'info', 'params');
    end
end

function eps_samples = generate_epsilon_samples_sphere_radial(Q, num_dirs, num_radii, eps_max)
%GENERATE_EPSILON_SAMPLES_SPHERE_RADIAL Uniform directions on sphere + radial levels.
%   Returns eps_samples of size Q x Ns (real-valued).
    if Q <= 0 || num_dirs <= 0 || num_radii <= 0 || eps_max < 0
        error('Invalid epsilon sampling parameters');
    end
    dirs = randn(Q, num_dirs);
    dirs = dirs ./ vecnorm(dirs);
    if num_radii == 1
        radii = 0;
    else
        radii = linspace(0, eps_max, num_radii);
    end
    Ns = num_dirs * num_radii;
    eps_samples = zeros(Q, Ns);
    idx = 1;
    for i = 1:num_radii
        r = radii(i);
        for d = 1:num_dirs
            eps_samples(:, idx) = r * dirs(:, d);
            idx = idx + 1;
        end
    end
end
