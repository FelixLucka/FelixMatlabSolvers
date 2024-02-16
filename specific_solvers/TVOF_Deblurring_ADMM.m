function [u, info] = TVOF_Deblurring_ADMM(A, AT, f_or_ATf, v, alpha, gamma, para)
%TVOF_DEBLURRING_ADMM uses ADMM to do image deblurring regularized by
% TV and an optical flow transport term
%
% DESCRIPTION:
%   TVOF_Deblurring_ADMM uses ADMM to do image deblurring regularized by
%   TV and an optical flow transport term almost as described in
%   "Enhancing Compressed Sensing Photoacoustic Tomography by Simultaneous
%   Motion Estimation" by Arridge, Beard, Betcke, Cox, Huynh, Lucka Zhang,
%   2018,
%   but in the above paper, only denoising was outlined.
%
%   The algorithm tries to minimize the following energy:
%   u = argmin { \sum_t^T  1/2 \| A_t u_t - f_t \|_2^2 + \alpha \|\nabla u_t \|_1
%                + \gamma^p/p \| rho(u_{t+1}, u_{t}, v_t) \|_p^p }
%   OR if symmetricOF = true:
%   u = argmin { \sum_t^T  1/2 \| A_t u_t - f_t \|_2^2 + \alpha_t \|\nabla u_t \|_1
%                + \gamma_t^p/(2*p) \| rho(u_{t+1}, u_t, v_t) \|_p^p
%                + \gamma_t^p/(2*p) \| rho(u_{t}, u_{t+1}, -v_t) \|_p^p}
%   where rho(u_{t+1}, u_t, v_t) = u_{t+1} - u_t + (\nabla u_{t+1}) \cdot v_t
%      or rho(u_{t+1}, u_t, v_t) = warp(u_{t+1} v_t) - u_t
%
%   For a general reference on ADMM, we refer to "Distributed Optimization
%   and Statistical Learning via the Alternating Direction Method of Multipliers"
%   by Boyd et al, 2011 and "An introduction to continuous optimization for
%   imaging" by Chambolle and Pock, 2016.
%
% USAGE:
%   [u, info] = TVOF_Deblurring_ADMM(A, AT, ATf, v, alpha, gamma, para)
%
% INPUTS:
%   A      - function handle to the forward operator
%   AT     - function handle to the adjoint of the forward operator
%   f_or_ATf - either the data f as cell array or AT(f) as 2D+1D or
%            3D+1D numerical array (needs to be specified in para)
%   v - 2D+1D+1D or 3D+1D+1D motion field
%   alpha - regularization parameter for spatial TV
%   gamma - regularization parameter for transport term
%
%   para  - struct containing additional parameters:
%     'useATF' -  logical indicating whether fOrATf contains f (false)
%                   or AT*f (true)
%     'dimU'  - dimension of u as [nX, nX, T] or [nX, nX, nZ, T]
%     'u' - inital value for u
%     'ADMMVar' - initial values for ADMM variables
%     'OFtype' - 'linear' or 'nonLinear' depending whether a linearization
%     of the optical flow equation in v is used or not (it is always linear
%     in u)
%     'TVtype' - total variation type used for motion field
%     'p'      - index of the Lp norm, see above
%     'symmetricOF' - boolean determining whether the OF part is
%       symmetric in u_1 and u_2 by adding two OF terms (see above, default: false)
%     'constraint' - constraint on image: 'none' (df), 'positivity' 'range'
%     'dt'      - the time differences between frames as a vector
%       (size T-1 x 1) which will be used to weight the temporal finite
%       differences everywhere.
%     'maxEval' - maximal number of operator evaluations
%     'stopCriterion' - choose ... to stop if ... :
%       'primalDualRes' - both primal and dual residuum are small
%                         enough (see Section 3.3.1. in Boyd et al.)
%       'energy' - a certain energy level has been reached
%       'maxEval' if a maximal number of operator evaluations is reached.
%       'stopTolerance' - stop tolerance (see above)
%     'typicalScale' - typical scale of v (for certain stop criteria)
%     'overRelaxPara' - over-relaxation parameter (see Section 3.4.3. in
%            Boyd et al, 2011), default: 1.8, i.e., overrelaxation is in use
%     'dataCast' - numerical precision and computing unit of main variables:
%                  'single','double' (df),'gpuArray-single','gpuArray-double'
%
%      'lsSolverPara' - struct with parameters for least-squares sub-problem:
%        'lsSolver' - 'CG' (df) or 'minres'
%        'lsMaxIter' - max iterations for LS solver (df 100)
%        'stopCriterion'  - 'maxIter','relRes' (df) or 'progRelRes': stop
%          criterion, 'progRelRes' uses a progressive tolerance for the
%          relative residual criterion
%        'progMode' - 'last' (df),'fac' or 'poly': way to decrease
%            tolerance for 'stopCriterion' = 'progRelRes': 'last' stops if
%            relRes is smaller than in the previous iteration; 'fac' multiplies the
%            stop tolerance with fixed factor; 'poly' uses tol_0 / k^tolDecExp
%        'tol' - tolerance for relative residuum
%        'tolDecFac' - factor   for 'progMode' = 'fac'  (df 0.99)
%        'tolDecExp' - exponent for 'progMode' = 'poly' (df 2)
%
%        'output' - boolean determining whether output should be displayed (df: true)
%        'outputPreStr' - pre-fix for output str (to structure output in multi-layered algos);
%        'outputFreq' - computational time in sec after which new output
%                       is displayed (to limit output in small-scale scenarios)
%        'returnAlgoVar'- boolean determining whether the auxillary variables of
%                         sub-routines should be returned to warm-start the sub-routines
%                         in the next call to TVOF_Deblurring_ADMM.m (df: true)
%         'rho'          - value of augmentation term (df: 10)
%         'rhoAdapt'     - boolean controlling whether rho should be adapted (df: false)
%         'rhoAdaptMode' - way to adapt rho :'normal','slow','averaged','energy'
%                           (undocumented)
%         'mu'  - rho adaptation parameter, see Section 3.4.1 in Boyd et al, 2011
%         'tau' - rho adaptation parameter, see Section 3.4.1 in Boyd et al, 2011
%         'rhoMin'        - lower bound for rho
%         'computeEnergy' - boolean determining whether energy should be
%                              computed (df: true)
%         'computeInitialEnergy' - boolean determining whether only the
%             energy of the initial guess should be computed and returned
%             (for compartibilty and precision reasons)
%
% OUTPUTS:
% 	u     - reconstructed images
% 	info  - struct which stores details and variables of the iteration,
%               can be used to inform subsequent calls to this function
%
% ABOUT:
% 	author          - Felix Lucka
% 	date            - 06.05.2018
% 	last update     - 27.10.2023
%
% see also ADMM.m, TV_Deblurring


clock_cmp_total = tic; % track total computation time


% =========================================================================
% CHECK INPUT AND INITILIZE VARIABLES
% =========================================================================


info = [];

%%% read out general algorithm parameters
use_ATf           = checkSetInput(para, 'useATF', 'logical', false);
max_eval          = checkSetInput(para,'maxEval','i,>0',1000);
max_iter          = ceil(max_eval/2);
stop_criterion    = checkSetInput(para, 'stopCriterion', ...
    {'primalDualRes','energy','maxEval'}, 'maxEval');
stop_tol          = checkSetInput(para, 'stopTolerance', '>=0', 10^-6);
typical_scale     = checkSetInput(para, 'typicalScale', '>=0', 1);
output            = checkSetInput(para, 'output', 'logical', false);
output_pre_str    = checkSetInput(para, 'outputPreStr', 'char', '');
output_freq       = checkSetInput(para, 'outputFreq', 'i,>0', 0);
return_ADMM_var   = checkSetInput(para, 'returnADMMVar', 'logical', true);
data_cast         = checkSetInput(para, 'dataCast', ...
    {'single','double','gpuArray-single','gpuArray-double'}, 'double');
rho               = checkSetInput(para, 'rho', '>0', 10);
rho_adaptation    = checkSetInput(para, 'rhoAdaptation', 'logical', false);
mu                = checkSetInput(para, 'mu', '>0', 10);
tau               = checkSetInput(para, 'tau', '>0', 2);
rho_min           = checkSetInput(para, 'rhoMin', '>0', 10^-3);
overrelax_para    = checkSetInput(para, 'overRelaxPara', '>0', 1.8);
cmp_energy        = checkSetInput(para, 'computeEnergy', 'logical', false);
cmp_ini_energy    = checkSetInput(para, 'computeIniEnergy', 'logical', false);


%%% read out dimensions
if(use_ATf)
    sqnorm_f = checkSetInput(para, 'sqNormF', '>0', 0);
    sz_u    = size(f_or_ATf);
    ATf    = f_or_ATf;
    %     [szF, df] = checkSetInput(para, 'szF', 'i,>0', 0);
    %     if(df)
    %         szF = cellfun(@size, A(fOrATf), 'UniformOutput', false);
    %     end
else
    % we're given f, we need to get the size of U through parameters or by
    % one application of A' * f
    %     if(iscell(fOrATf))
    %         szF       = cellfun(@size, fOrATf, 'UniformOutput', false);
    %     else
    %         szF       = size(fOrATf);
    %     end
    ATf = AT(f_or_ATf);
    sz_u = size(ATf);
    if(iscell(f_or_ATf))
        sqnorm_f = sum(cellfun(@(x) sum(x(:).^2), f_or_ATf));
    else
        sqnorm_f = sum(f_or_ATf(:).^2);
    end
end
n          = prod(sz_u);
%nF         = sum(cellfun(@prod, szF));
dim_space   = length(sz_u(1:end-1));
n_xyz       = sz_u(1:dim_space);
n_t         = sz_u(end);
sz_grad_u   = [n_xyz dim_space n_t];
sz_trs      = sz_u;
sz_trs(end) = n_t - 1;

%%% initialize iteration variables
[castFun, castZeros] = castFunction(data_cast);
[u, ~, para] = checkSetInput(para, 'u', 'numeric', castZeros(sz_u), [], true);

[ADMM_var, ~, para] = checkSetInput(para, 'ADMMVar', 'struct', emptyStruct, [], true);

%%% cast all variables to the desired type
cast_variables = {'u','v','f_or_ATf', 'rho','overrelax_para','alpha','gamma','sqnorm_f'};
% if(output)
%     disp([outputPreStr 'casting variables to type: ' dataCast])
% end
for cast_index = 1:length(cast_variables)
    eval([cast_variables{cast_index} ' = castFun(' cast_variables{cast_index} ');']);
end


%%% read out parameters defining the objective function and initial values
if(isscalar(alpha))
    alpha = alpha * ones(n_t, 1);
end
alpha      = alpha(:);
alpha_exp   = reshape(alpha, [ones(1, dim_space+1) n_t]);
if(isscalar(gamma))
    gamma = gamma * ones(n_t-1, 1);
end
gamma      = gamma(:);
gamma_exp   = reshape(gamma, [ones(1, dim_space) n_t-1]);

%%% for the split variables and lagrange multipliers
OF_type      = checkSetInput(para, 'OFtype', {'linear','nonLinear'}, 'linear');
TV_type      = checkSetInput(para, 'TVtype', {'isotropic'}, 'isotropic');
p            = checkSetInput(para, 'p', '>0', 2);
symmetric_OF = checkSetInput(para, 'symmetricOF', 'logical', false);
constraint   = checkSetInput(para, 'constraint', {'none','nonNegative', 'range'},'none');
OF_para.dt   = checkSetInput(para, 'dt', '>0', ones(1, n_t-1));
OF_para.warpPara = checkSetInput(para, 'warpPara', 'struct', emptyStruct);
OF_para.dimSpace = dim_space;

[z_grad, ~, ADMM_var]   = checkSetInput(ADMM_var, 'zGrad', 'numeric', castZeros(sz_grad_u), [], true);
z_grad = castFun(z_grad);
if(~isfield(ADMM_var, 'wGrad') && any(u(:)))
    w_grad_df = spatialFwdGrad(u, false) - z_grad;
else
    w_grad_df = - z_grad;
end
[w_grad, ~, ADMM_var] = checkSetInput(ADMM_var, 'wGrad', 'numeric', w_grad_df, true);
clear w_grad_df
w_grad = castFun(w_grad);

switch p
    case 2
        % nothing to be done
    otherwise
        [z_trs, ~, ADMM_var] = checkSetInput(ADMM_var, 'zTrs', 'numeric', castZeros(sz_trs), [], true);
        z_trs               = castFun(z_trs);
        if(~isfield(ADMM_var, 'wTrs') && any(u(:)))
            w_trs_df = opticalFlowOperator(u, v, OF_type, false, OF_para) - z_trs;
        else
            w_trs_df = -z_trs;
        end
        [w_trs, ~, ADMM_var] = checkSetInput(ADMM_var, 'wTrs', 'numeric', w_trs_df, [], true);
        clear w_trs_df
        w_trs               = castFun(w_trs);

        if(symmetric_OF)
            [z_trs_twist, ~, ADMM_var] = checkSetInput(ADMM_var, 'zTrsTwist', 'numeric', castZeros(sz_trs), [], true);
            z_trs_twist               = castFun(z_trs_twist);
            if(~isfield(ADMM_var, 'wTrsTwist') && any(u(:)))
                w_trs_df = opticalFlowOperator(u, v, OF_type, false, OF_para, true) - z_trs_twist;
            else
                w_trs_df = -z_trs_twist;
            end
            [w_trs_twist, ~, ADMM_var] = checkSetInput(ADMM_var, 'wTrsTwist', 'numeric', w_trs_df, [], true);
            clear w_trs_df
            w_trs_twist               = castFun(w_trs_twist);
        end
end


%%% box constraints on u
switch constraint
    case 'none'
        con_range = [];
        con_u     = false;
    case 'nonNegative'
        con_range = [];
        con_u     = true;
    case 'range'
        con_range = checkSetInput(para, 'conRange', 'double', [], 'error');
        if(~ length(con_range) == 2 && con_range(1) < con_range(2))
            error(['invalid range: ' num2str(con_range)])
        end
        con_u     = true;
end
projectionU  = @(u) getfield(projBoxConstraints(u, constraint, con_range), 'x');


%%% set up least squares solvers
ls_solver_df = 'CG';
if(con_u)
    split_con_var = checkSetInput(para,'splitConVar', 'logical', true);
    return_split_con_var = checkSetInput(para,'returnSplitConVar', 'logical', false);
    if(split_con_var)
        con_violation            = castZeros([max_iter+1, 1]);
        con_violation(1)         = maxAll(abs(u - projectionU(u)));
        min_con_violation        = con_violation(1);
        last_output_con_violation = min_con_violation;
        %%% TODO: a value not 1 does not seem to work
        reg_con                 = checkSetInput(para, 'regCon', '>0', 1);

        [z_con, ~, ADMM_var] = checkSetInput(ADMM_var, 'zCon', 'numeric', castZeros(sz_u), [], true);
        z_con               = castFun(z_con);
        w_con               = checkSetInput(ADMM_var, 'wCon', 'numeric', projectionU(u), true);
        w_con               = castFun(w_con);
    else
        ls_solver_df = 'L-BFGS-B';
        reg_con     = 1;
    end
else
    reg_con = 1;
end


%%% read out least squares parameters
ls_solver_para  = checkSetInput(para, 'lsSolverPara', 'struct', emptyStruct);
ls_solver       = checkSetInput(ls_solver_para, 'lsSolver', {'CG','minres','bicgstab','L-BFGS-B'}, ls_solver_df);
ls_max_iter     = checkSetInput(ls_solver_para, 'maxIter', 'i,>0', 1000);
ls_min_iter     = checkSetInput(ls_solver_para, 'minIter', 'i,>0', 1);
ls_stop_criterion = checkSetInput(ls_solver_para, 'stopCriterion', ...
    {'maxIter','relRes','progRelRes'}, 'relRes');
switch ls_stop_criterion
    case 'progRelRes'
        ls_prog_mode = checkSetInput(ls_solver_para, 'progMode', {'last','fac','poly'}, 'last');
        switch ls_prog_mode
            case 'last'
                ls_tol = checkSetInput(ls_solver_para, 'tol', '>0', 10^-4);
            case 'fac'
                ls_tol = checkSetInput(ls_solver_para, 'tol', '>0', 10^-4);
                ls_tol_dec_fac = checkSetInput(ls_solver_para, 'tolDecFac', '>0', 0.99);
            case 'poly'
                ls_tol    = checkSetInput(ls_solver_para, 'tol', '>0', 10^-4);
                ls_tol_ini = ls_tol;
                ls_tol_dec_exp    = checkSetInput(ls_solver_para, 'tolDecExp', '>0', 2);
                ls_tol_dec_offset = checkSetInput(ls_solver_para, 'tolDecOffset', '>=0', 0);
        end
    case 'relRes'
        ls_tol = checkSetInput(ls_solver_para, 'tol', '>0', 10^-6);
    case 'maxIter'
        ls_tol = checkSetInput(ls_solver_para, 'tol', '>0', eps(castFun(1)));
end
info.cgIter = zeros(max_iter, 1);


%%% initialize variables for the least squares solvers
ATA = checkSetInput(para, 'ATA', 'function_handle', @(u) AT(A(u)));
switch ls_solver
    case 'CG'
        if(any(u(:)))
            %             [GGu,dfFL] = checkSetInput(ADMMVar,'GGu','numeric',1);
            %             if(dfFL)
            GGu = GGfun(u, reg_con);
            %             else
            %                GGu = castFun(GGu);
            %             end
        else
            GGu = u;
        end
    case {'minres', 'bicgstab'}
        % nothing to be done here
    case 'L-BFGS-B'
        LBFGS_opt = checkSetInput(ls_solver_para, 'LBFGSopt', 'struct', emptyStruct);
        LBFGS_opt.maxIts     = ls_max_iter;
        LBFGS_opt.m          = checkSetInput(LBFGS_opt, 'm', 'i,>0', 6);
        LBFGS_opt.pgtol      = 0;
        LBFGS_opt.factr      = 0;
        LBFGS_opt.printEvery = Inf;
end
rel_res_ls = Inf;

%%% initialize energy and stop value variables
stop_value          = Inf;
stop                = false;
cmp_primal_dual_res = true;
switch stop_criterion
    case 'primalDualRes'
        cmp_primal_dual_res = true;
        eps_abs             = typical_scale * stop_tol;
        eps_rel             = stop_tol;
        stop_tol            = 1;
    case 'energy'
        cmp_energy         = true;
end
cmp_primal_dual_res = rho_adaptation || cmp_primal_dual_res;
info       = [];
iter       = 0;
n_eval     = 0;
eval_vs_iter = zeros(max_iter, 1);


% set up energy functions
if(cmp_energy)

    if(use_ATf)
        dataFitFunAux = @(Au, u) (sumAll(Au .* Au) - 2 * sumAll(u .* f_or_ATf) + sqnorm_f);
        dataFitFun = @(u) 0.5 * dataFitFunAux(double(A(u)), double(u));
    else
        if(iscell(f_or_ATf))
            dataFitFun = @(u) 0.5 * sum(cellfun(@(x,y) sumAll((x-y).^2), A(u), f_or_ATf));
        else
            dataFitFun = @(u) 0.5 * sum(vec(A(u) - f_or_ATf).^2);
        end
    end

    energy          = castZeros([max_iter+1, 1]);
    [energy(1), df] = checkSetInput(para, 'energy', '>0', 1);
    if(df)
        if(any(u(:)))
            datafit = dataFitFun(u);
            TVu     = sum(alpha .* sumAllBut(sqrt(sum(spatialFwdGrad(u,false).^2, dim_space+1)), 'last'));
            OFuv    = sum(gamma.^p/p .* sumAllBut(abs(opticalFlowOperator(u, v, OF_type, false, OF_para)).^p, 'last'));
            if(symmetric_OF)
                OFuv    = 0.5 * (OFuv +  sum(gamma.^p/p .* sumAllBut(abs(opticalFlowOperator(u, v, OF_type, false, OF_para, true )).^p, 'last')));
            end
        else
            datafit     = 0.5 * sqnorm_f;
            TVu         = castZeros([1,1]);
            OFuv        = castZeros([1,1]);
            energy(1)   = datafit;
        end
        energy(1)       = datafit + TVu + OFuv;
    end
    min_energy         = energy(1);
    last_output_energy = energy(1);

    if(cmp_ini_energy)
        info.energy     = gather(energy);
        info.minEnergy  = gather(min_energy);
        info.dataFit    = gather(datafit);
        info.TVu        = gather(TVu);
        info.optFlwDisc = gather(OFuv);
        return
    end
end


%%% clear parameter structs
clear para ADMM_var


%%% display output
if(output)
    output_str = output_pre_str;
    output_str = [output_str 'Starting TV + OF minimization using ADMM'];
    if(cmp_energy)
        output_str = [output_str '; energy: ' num2str(min_energy, '%.4e')];
        last_output_energy = min_energy;
    end
    next_eval_output = output_freq;
    disp(output_str)
end


% =========================================================================
% MAIN ITERATION
% =========================================================================


while(~stop && n_eval < max_eval)


    %%% proceed with the iteration
    iter = iter + 1;

    %%% use least-squares solver to update u
    switch ls_solver
        case 'CG'
            CGsolve()
        case 'minres'
            minresSolve()
        case 'bicgstab'
            bicgstabSolve()
        case 'L-BFGS-B'
            LBFGSsolve()
    end
    if(cmp_energy)
        datafit = dataFitFun(u);
    end

    % modify least-squares stop criterion
    switch ls_stop_criterion
        case 'progRelRes'
            switch ls_prog_mode
                case 'fac'
                    ls_tol = ls_tol_dec_fac * ls_tol;
                case 'poly'
                    ls_tol = ls_tol_ini/((iter + ls_tol_dec_offset)^ls_tol_dec_exp);
                case 'last'
                    ls_tol = rel_res_ls;
            end
            output_ls = ['iLS: ' int2str(info.lsIter(iter)) ', rrLS: ' num2str(rel_res_ls,'%.2e')];
        case 'maxIter'
            output_ls = ['rrLS: ' num2str(rel_res_ls, '%.2e')];
        case 'relRes'
            output_ls = ['rrLS: ' int2str(info.lsIter(iter))];
    end


    %%% update the different split variables v and dual variables w

    % gradient
    Eu_grad = spatialFwdGrad(u, false);
    if(cmp_energy)
        TVu     = sum(alpha .* sumAllBut(sqrt(sum(Eu_grad.^2, dim_space+1)), 'last'));
    end
    % over-relaxation
    aux_var_grad = overrelax_para * Eu_grad + (1-overrelax_para) * z_grad; % F(zGrad) = - zGrad
    % update zGrad
    z_grad_old  = z_grad;
    z_grad      = getfield(proxL21(aux_var_grad + w_grad, alpha_exp/rho, dim_space+1), 'x');

    % update w
    w_grad    = w_grad + (aux_var_grad - z_grad); % F(zGrad) = - zGrad
    if(cmp_primal_dual_res)
        % compute primal and dual residuum
        primal_res_norm = sqrt(sumAll((Eu_grad - z_grad).^2)) / (2*n);
        dual_res        = spatialFwdGrad(z_grad - z_grad_old, true);
    end
    clear Eu_grad aux_var_grad z_grad_old

    % optical flow
    switch p
        case 2
            if(cmp_energy)
                OFuv    = sum(gamma.^p/p .* sumAllBut((opticalFlowOperator(u, v, OF_type, false, OF_para, false)).^2, 'last'));
                if(symmetric_OF)
                    OFuv = 0.5 * (OFuv + sum(gamma.^p/p .* sumAllBut((opticalFlowOperator(u, v, OF_type, false, OF_para, true)).^2, 'last')));
                end
                energy_u = datafit + TVu + OFuv;
            end
        otherwise

            Eu_trs = opticalFlowOperator(u, v, OF_type, false, OF_para, false);

            if(cmp_energy)
                OFuv    = sum(gamma./p .* sumAllBut(abs(Eu_trs).^p, 'last'));
                energy_u = datafit + TVu + OFuv;
            end
            % overrelaxation
            aux_var_trs = overrelax_para * Eu_trs + (1-overrelax_para) * z_trs; % F(zTrs) = - zTrs
            % update vTrs
            z_trs_old   = z_trs;
            if(symmetric_OF)
                z_trs      = getfield(proxLp(aux_var_trs + w_trs, 0.5 * gamma_exp.^p/rho, p), 'x');
            else
                z_trs      = getfield(proxLp(aux_var_trs + w_trs, gamma_exp.^p/rho, p), 'x');
            end

            % update w
            w_trs      = w_trs + (aux_var_trs - z_trs);
            if(cmp_primal_dual_res)
                % compute primal and dual residuum
                primal_res_norm = primal_res_norm + sqrt(sumAll((Eu_trs - z_trs).^2)) / n;
                dual_res       = dual_res + opticalFlowOperator(z_trs-z_trs_old,v, OF_type, true, OF_para);
            end
            clear Eu_trs aux_var_trs z_trs_old

            if(symmetric_OF)

                Eu_trs_twist = opticalFlowOperator(u, v, OF_type, false, OF_para, true);

                if(cmp_energy)
                    OFuv    = 0.5 * (OFuv + sum(gamma./p .* sumAllBut(abs(Eu_trs_twist).^p, 'last')));
                    energy_u = datafit + TVu + OFuv;
                end
                % overrelaxation
                aux_var_trs_twist = overrelax_para * Eu_trs_twist + (1-overrelax_para) * z_trs_twist; % F(zTrs) = - zTrs
                % update zTrs
                z_trs_old_twist   = z_trs_twist;
                z_trs_twist      = getfield(proxLp(aux_var_trs_twist + w_trs_twist, 0.5 * gamma_exp.^p/rho, p), 'x');

                % update w
                w_trs_twist      = w_trs_twist + (aux_var_trs_twist - z_trs_twist);
                if(cmp_primal_dual_res)
                    % compute primal and dual residuum
                    primal_res_norm = primal_res_norm + sqrt(sumAll((Eu_trs_twist - z_trs_twist).^2)) / n;
                    dual_res       = dual_res + opticalFlowOperator(z_trs_twist-z_trs_old_twist,v, OF_type, true, OF_para, true);
                end
                clear Eu_trs_twist aux_var_trs_twist z_trs_old_twist

            end
    end


    % box constraints
    if(con_u && split_con_var)
        % overrelaxation
        aux_var_con = overrelax_para * reg_con * u + (1-overrelax_para) * z_con; % EuCon =  regCon * u; F(zCon) = - zCon
        % update zCon
        z_con_old = z_con;
        z_con  = projectionU(aux_var_con + w_con);
        % update w
        w_con    = w_con + (aux_var_con - z_con);
        if(cmp_primal_dual_res)
            % compute primal and dual residuum
            primal_res_norm = primal_res_norm + sqrt(sumAll((reg_con * u - z_con).^2)) / n;
            dual_res = dual_res + reg_con * (z_con-z_con_old);
        end
        clear aux_var_con
        con_violation(iter+1) = maxAll(abs(u - projectionU(u)));
        if(con_violation(iter+1) < con_violation(iter))
            min_con_violation = con_violation(iter+1);
        end
    end

    % compute dual norm
    if(cmp_primal_dual_res)
        dual_res_norm    = rho*sqrt(sumAll(dual_res.^2)) / n;
        clear dual_res
    end

    %%% book-keeping
    n_eval             = n_eval + 1;
    eval_vs_iter(iter+1) = n_eval;

    if(cmp_energy)
        energy(iter+1) = energy_u;
        if(energy(iter+1) < min_energy)
            min_energy = energy(iter + 1);
        elseif(con_u && split_con_var && con_violation(iter + 1) < con_violation(iter))
            % constraints got better,
            min_energy = energy(iter + 1);
        end
    end



    %%% update rho
    if(rho_adaptation)
        if(primal_res_norm > mu * dual_res_norm)
            rho = tau * rho;
            w_grad = w_grad/tau;
            if(p ~= 2)
                w_trs = w_trs/tau;
                if(symmetric_OF)
                    w_trs_twist = w_trs_twist/tau;
                end
            end
            if(con_u && split_con_var)
                w_con = w_con/tau;
            end
            if(exist('GGu','var'))
                GGu = GGfun(u, reg_con);
            end
            n_eval = n_eval + 2;
        elseif(dual_res_norm > mu * primal_res_norm && rho/tau > rho_min)
            rho = rho/tau;
            w_grad = tau * w_grad;
            if(p ~= 2)
                w_trs = tau * w_trs;
                if(symmetric_OF)
                    w_trs_twist = tau * w_trs_twist;
                end
            end
            if(con_u && split_con_var)
                w_con = tau * w_con;
            end
            if(exist('GGu','var'))
                GGu = GGfun(u, reg_con);
            end
            n_eval = n_eval + 2;
        end
    end


    %%% plotting and output
    if(output && n_eval >= next_eval_output)
        output_str =  [output_pre_str 'it ' int2str(iter)];
        output_str =  [output_str ', evl ' int2str(n_eval) '/' int2str(max_eval)];
        output_str =  [output_str  '; ' output_ls];
        switch stop_criterion
            case 'maxEval'
                if(cmp_primal_dual_res)
                    output_str = [output_str  '; pr/du res: '  ...
                        num2str(primal_res_norm, '%.2e') '/' num2str(dual_res_norm, '%.2e')];
                end
                if(rho_adaptation)
                    output_str = [output_str  '; rho: '  num2str(rho, '%.2e')];
                end

                if(cmp_energy)
                    output_str = [output_str '; en: ' num2str(energy(iter+1), '%.2e') ...
                        ' (dFac: ' num2str((last_output_energy-energy(iter+1))/last_output_energy, '%.2e') ')'];
                    if(energy(iter+1) > min_energy)
                        output_str = [output_str '*'];
                    end
                    last_output_energy = energy(iter+1);
                end
                if(con_u && split_con_var)
                    output_str = [output_str '; conV: ' num2str(con_violation(iter+1), '%.2e')];
                    if(con_violation(iter+1) > min_con_violation)
                        output_str = [output_str '*'];
                    end
                    last_output_con_violation = con_violation(iter+1);
                end
            case {'primalDualRes','energy'}
                notImpErr
        end
        disp(output_str)
        next_eval_output = n_eval + output_freq;
    end

end



% =========================================================================
% GATHER RESULTS AND CLEAN UP
% =========================================================================


u = gather(u);

info.rho               = gather(rho);
info.evalVsIter        = eval_vs_iter(1:iter+1);
if(return_ADMM_var)
    info.ADMMVar.zGrad = gather(z_grad); clear z_grad
    info.ADMMVar.wGrad = gather(w_grad); clear w_grad
    if(p ~= 2)
        info.ADMMVar.zTrs = gather(z_trs); clear z_trs
        info.ADMMVar.wTrs = gather(w_trs); clear w_trs
        if(symmetric_OF)
            info.ADMMVar.zTrsTwist = gather(z_trs_twist); clear z_trs_twist
            info.ADMMVar.wTrsTwist = gather(w_trs_twist); clear w_trs_twist
        end
    end
    if(con_u && split_con_var)
        % return split variable instead of u
        if(return_split_con_var)
            u = gather(z_con) / reg_con;
        end
        info.ADMMVar.zCon = gather(z_con); clear z_con
        info.ADMMVar.wCon = gather(w_con); clear w_con
    end

    switch ls_solver
        case 'CG'
            % info.ADMMVar.GGu = gather(GGu);
    end
else
    info.ADMMVar = emptyStruct;
end

if(cmp_energy)
    info.energy     = gather(energy(1:iter+1));
    info.minEnergy  = gather(min_energy);
    info.dataFit    = gather(datafit);
    info.TVu        = gather(TVu);
    info.optFlwDisc = gather(OFuv);
end

if(con_u && split_con_var)
    info.conViolation = gather(con_violation(1:iter+1));
end

info.lsSolverPara = emptyStruct;
switch ls_stop_criterion
    case 'progRelRes'

        switch ls_prog_mode
            case 'poly'
                info.lsSolverPara.tolDecOffset = ls_tol_dec_offset + iter;
            case {'fac', 'last'}
                info.lsSolverPara.tol = ls_tol;

        end
end

info.tCompTotal = toc(clock_cmp_total);


% =========================================================================
% NESTED FUNCTIONS
% =========================================================================


%%% compute G * v for the linear operator G in the least-squares problem
    function res = Gfun(uIn, alpha_here, gamma_here, reg_con_here, adjoint_fl)
        oldImpErr
        if(~adjoint_fl)
            res = vec(A(reshape(uIn,sz_u)))/sqrt(rho);
            switch p
                case 2
                    res = [res; sqrt(gamma_here^p/rho) * vec(opticalFlowOperator(reshape(uIn, sz_u), v, OF_type, false, OF_para, false))];
                    if(symmetric_OF)
                        res = [res; sqrt(gamma_here^p/rho) * vec(opticalFlowOperator(reshape(uIn, sz_u), v, OF_type, false, OF_para, true))];
                    end
                otherwise
                    res = [res; gamma_here * vec(opticalFlowOperator(reshape(uIn, sz_u), v, OF_type, false, OF_para, false))];
                    if(symmetric_OF)
                        res = [res; gamma_here * vec(opticalFlowOperator(reshape(uIn, sz_u), v, OF_type, false, OF_para, true))];
                    end
            end
            res = [res(:); alpha_here * vec(spatialFwdGrad(reshape(uIn, sz_u), false))];
            if(con_u && split_con_var)
                res = [res; reg_con_here * uIn(:)];
            end
        else
            res = vec(AT(reshape(uIn(1:nF), szF)))/sqrt(rho);
            switch p
                case 2
                    res = res + sqrt(gamma_here^p/rho) * vec(opticalFlowOperator(reshape(uIn(nF+1:(nF+n)),sz_trs), v, OF_type, true, OFPara));
                otherwise
                    res = res + gamma_here * vec(opticalFlowOperator(reshape(uIn(nF+1:(nF+n)),sz_trs), v, OFTpye, true, OFPara));
            end
            res = res + alpha_here * vec(spatialFwdGrad(reshape(uIn(nF+n+1:((1+dim_space)*n+nF)),sz_grad_u),true));
            if(con_u && split_con_var)
                res = res + reg_con_here * uIn(((1+dim_space)*n+nF)+1:end);
            end
        end
    end


%%% compute G' G * v for the linear operator G in the least-squares problem
    function res = GGfun(u_in, reg_con_here)

        %%% start with the transport term
        %         switch OFtype
        %             case 'linear'
        %                 % forward difference in time
        %                 aux = tempFwdDiff(uIn, false, OFpara.dt);
        %
        %                 % central differences in space
        %                 if(any(v(:)))
        %                     switch dimSpace
        %                         case 2
        %                             aux = aux + v(:,:,:,1) .* specialCenDiff(uIn(:,:,2:end), 'Hen', 1) + ...
        %                                 v(:,:,:,2) .* specialCenDiff(uIn(:,:,2:end),'Hen', 2);
        %                         case 3
        %                             aux = aux + v(:,:,:,:,1) .* specialCenDiff(uIn(:,:,:,2:end), 'Hen',1) + ...
        %                                 v(:,:,:,:,2) .* specialCenDiff(uIn(:,:,:,2:end), 'Hen', 2) + ...
        %                                 v(:,:,:,:,3) .* specialCenDiff(uIn(:,:,:,2:end), 'Hen', 3);
        %                         otherwise
        %                             notImpErr
        %                     end
        %                 end
        %                 % add adjoint of transport term applied to w
        %                 % adjoint forward difference in time
        %                 res = tempFwdDiff(aux, true, OFpara.dt);
        %                 % adjoint central differences in space
        %                 if(any(v(:)))
        %                     switch dimSpace
        %                         case 2
        %                             res = res - specialCenDiff(v(:,:,:,1) .* aux, 'HenAdj', 1);
        %                             res = res - specialCenDiff(v(:,:,:,2) .* aux, 'HenAdj', 2);
        %                         case 3
        %                             res = res - specialCenDiff(v(:,:,:,:,1) .* aux, 'HenAdj', 1);
        %                             res = res - specialCenDiff(v(:,:,:,:,2) .* aux, 'HenAdj', 2);
        %                             res = res - specialCenDiff(v(:,:,:,:,3) .* aux, 'HenAdj', 3);
        %                         otherwise
        %                             notImpErr
        %                     end
        %                 end
        %             case 'nonLinear'
        res = opticalFlowOperator(u_in, v, OF_type, false, OF_para, false);
        switch p
            case 2
                res = bsxfun(@times, gamma_exp.^p/rho , res);
                if(symmetric_OF)
                    res = 0.5 * res;
                end
            otherwise
                %res = res;
        end
        res = opticalFlowOperator(res, v, OF_type, true, OF_para, false);

        if(symmetric_OF)
            res_twist = opticalFlowOperator(u_in, v, OF_type, false, OF_para, true);
            switch p
                case 2
                    res_twist = bsxfun(@times, 0.5 * gamma_exp.^p/rho , res_twist);
                otherwise
                    %resTwist = resTwist;
            end
            res_twist = opticalFlowOperator(res_twist, v, OF_type, true, OF_para, true);
            res = res + res_twist;
        end

        %%% add the spatial laplace operator
        aux = zeros(size(u_in), 'like', u_in);
        switch dim_space
            case 2
                % x part
                aux(1,:,:)       = u_in(1,:,:) - u_in(2,:,:);
                aux(2:end-1,:,:) = -diff(u_in, 2, 1);
                aux(end,:,:)     = u_in(end,:,:) - u_in(end-1,:,:);
                res              = res + aux;
                % y part
                aux(:,1,:)       = u_in(:,1,:) - u_in(:,2,:);
                aux(:,2:end-1,:) = -diff(u_in, 2, 2);
                aux(:,end,:)     = u_in(:,end,:) - u_in(:,end-1,:);
                res              =  res + aux;
            case 3
                % x part
                aux(1,:,:,:)       = u_in(1,:,:,:) - u_in(2,:,:,:);
                aux(2:end-1,:,:,:) = -diff(u_in, 2, 1);
                aux(end,:,:,:)     = u_in(end,:,:,:) - u_in(end-1,:,:,:);
                res                =  res + aux;
                % y part
                aux(:,1,:,:)       = u_in(:,1,:,:) - u_in(:,2,:,:);
                aux(:,2:end-1,:,:) = -diff(u_in, 2, 2);
                aux(:,end,:,:)     = u_in(:,end,:,:) - u_in(:,end-1,:,:);
                res                =  res + aux;
                % z part
                aux(:,:,1,:)       = u_in(:,:,1,:) - u_in(:,:,2,:);
                aux(:,:,2:end-1,:) = -diff(u_in, 2, 3);
                aux(:,:,end,:)     = u_in(:,:,end,:) - u_in(:,:,end-1,:);
                res                =  res + aux;
            otherwise
                notImpErr
        end
        clear aux

        %%% add identity for box constraints
        if(con_u && split_con_var)
            res = res + reg_con_here^2 * u_in;
        end

        %%% finally, apply AT(A(u))
        res = res + ATA(u_in) / rho;
    end

%%% solve constrained least-squares problem by L-BFGS
    function LBFGSsolve()
        oldImpErr
        if(ls_min_iter > 1)
            notImpErr
        end

        % set up right hand side
        rhs_LBFGS = [f_or_ATf(:)/sqrt(rho); zeros(n,1); z_grad(:)-w_grad(:)];
        switch OF_type
            case 'linear'
                switch p
                    case 2
                        % nothing to be done
                    otherwise
                        rhs_LBFGS = [rhs_LBFGS; z_trs(:)-w_trs(:)];
                        if(symmetric_OF)
                            rhs_LBFGS = [rhs_LBFGS; z_trs_twist(:)-w_trs_twist(:)];
                        end
                end
            case 'nonLinear'
                notImpErr
        end

        LBFGS_opt.x0 = u(:);
        lb          = -inf(n,1);
        ub          =  inf(n,1);

        if(con_u && split_con_var)
            rhs_LBFGS = [rhs_LBFGS; sqrt(reg_con) * (z_con(:)-w_con(:))];
        else
            switch constraint
                case 'none'

                case 'non-negative'
                    lb  = zeros(n,1);
                case 'range'
                    lb  = ones(n,1) * con_range(1);
                    ub  = ones(n,1) * con_range(2);
            end
        end

        n_eval = n_eval + 1;
        res_sqnorm = [];
        info_LBFGS = [];
        evalc('[u, res_sqnorm, info_LBFGS] = lbfgsb(@(u_in) fgLBFGS(u_in,rhs_LBFGS), lb, ub, LBFGS_opt);'); % Perform minimization
        rel_res_ls          = sqrt(2*res_sqnorm)/norm(rhs_LBFGS(:));
        n_eval             = n_eval + 2*info_LBFGS.totalIterations;
        info.lsIter(iter) = info_LBFGS.totalIterations;
        u = reshape(u, sz_u);
    end

%%% function handle needed by L-BFGS
    function [fun_LBFGS, grad_LBGFS] = fgLBFGS(u_in, rhs_LBFGS)
        oldImpErr
        residual = Gfun(u_in,1,1,1,false) - rhs_LBFGS;
        fun_LBFGS = 0.5 * sum(residual(:).^2);
        grad_LBGFS = Gfun(residual,1,1,1,true);
    end

%%% update u by solving G'G u = G'c via CG
    function CGsolve()
        %%% update u by solving G'G u = G'c via CG
        % set up right hand side of normal equations
        r_cg = setupRHS();

        n_eval   = n_eval + 1;
        norm_rgs = sqrt(sum(r_cg(:).^2));
        r_cg     = r_cg - GGu;

        rho_cg        = sumAll(r_cg .* r_cg);
        min_relres_cg = sqrt(rho_cg) / norm_rgs;
        u_min_relres_cg = u;
        p_cg            = r_cg;
        for i_cg = 1:ls_max_iter
            q_cg      = GGfun(p_cg, reg_con);
            alpha_cg  = rho_cg/sumAll(p_cg .* q_cg);
            u        = u + alpha_cg * p_cg;
            GGu      = GGu + alpha_cg * q_cg;
            r_cg      = r_cg - alpha_cg * q_cg;
            rho_new_cg = sumAll(r_cg .* r_cg);
            relres_cg = sqrt(rho_new_cg) / norm_rgs;
            if(relres_cg < min_relres_cg)
                u_min_relres_cg = u;
                min_relres_cg  = relres_cg;
            end
            if(i_cg >= ls_min_iter && relres_cg < ls_tol && ~strcmp(ls_stop_criterion,'maxIter'))
                break
            end
            p_cg      = r_cg + (rho_new_cg / rho_cg) * p_cg;
            rho_cg    = rho_new_cg;
        end
        u                 = u_min_relres_cg;
        info.lsIter(iter) = i_cg;
        n_eval             = n_eval + 2*i_cg;
        rel_res_ls          = relres_cg;
        clear *_cg
    end

%%% set up the right hand side of the normal equations G'G u = G' * c
    function rhs = setupRHS()

        rhs     = ATf/rho + spatialFwdGrad(z_grad-w_grad, true);

        switch p
            case 2
                %
            otherwise
                rhs = rhs + opticalFlowOperator(z_trs-w_trs, v, OF_type, true, OF_para, false);
                if(symmetric_OF)
                    rhs = rhs + opticalFlowOperator(z_trs_twist-w_trs_twist, v, OF_type, true, OF_para, true);
                end
        end

        if(con_u && split_con_var)
            rhs = rhs + reg_con * (z_con-w_con);
        end

    end

%%% update u by solving G'G u = G'c via minRes
    function minresSolve()
        rhs = vec(setupRHS());
        GG_fun = @(x) vec(GGfun(reshape(x, sz_u), reg_con));
        [u, ~, rel_res_ls, i_solver, ~, ~] = ...
            minres(GG_fun, rhs, ls_tol, ls_max_iter, [], [], u(:));
        if(i_solver < ls_min_iter)
            [u, ~, rel_res_ls, ~, ~, ~] = ...
                minres(GG_fun, rhs(:), 0, ls_min_iter-i_solver, [], [], u(:));
            i_solver  = ls_min_iter;
        end
        n_eval = n_eval + 2 * i_solver;
        u     = reshape(real(u), sz_u);
        info.lsIter(iter) = i_solver;
    end

%%% update u by solving G'G u = G'c via bicgstab (usefull when G' does not match with G
%%% and G'G is not symmetric
    function bicgstabSolve()
        rhs = vec(setupRHS());
        GG_fun = @(x) vec(GGfun(reshape(x, sz_u), reg_con));
        [u, ~, rel_res_ls, i_solver] = ...
            bicgstab(GG_fun, rhs, ls_tol, ls_max_iter, [], [], u(:));
        if(i_solver < ls_min_iter)
            [u, ~, rel_res_ls] = ...
                bicgstab(GG_fun, rhs(:), 0, ls_min_iter-i_solver, [], [], u(:));
            i_solver  = ls_min_iter;
        end
        n_eval = n_eval + 2 * i_solver;
        u      = reshape(real(u), sz_u);
        info.lsIter(iter) = i_solver;
    end

end
