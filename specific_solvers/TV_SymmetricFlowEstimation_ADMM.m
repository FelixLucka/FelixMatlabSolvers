function [v_best, z, w, info] = TV_SymmetricFlowEstimation_ADMM(a1, a2, f1, f2, v, z, w, lambda, para)
%TV_FLOWESTIMATION_ADMM solves a TV regularized flow estimation problem by ADMM 
%
% DETAILS:
%   TV_FlowEstimation_ADMM.m solves this optimization problem 
%   min_v 1/p \| a1 .* v - f1 \|_p^p + 1/p \| a2 .* v - f2 \|_p^p + lambda * TV(v) 
%   using ADMM
%
% USAGE:
% 	[v_best, z, w, info] = TV_FlowEstimation_ADMM(a, f, v, z, w, lambda, para)
%
% INPUTS:
% 	a1/a2 - 2D/3D vector field (size [n_x, n_y, 2] or [n_x, n_y, n_z, 3])
%   f1/f2 - 2D/3D vector field (size a1/a2, v)
%   v  - intial value for motion field (size [n_x, n_y, 2] or [n_x, n_y, n_z, 3])
%   z  - initial value for split variable (size v)
%   w  - initial value for lagrange multipliers (size v)
%   lambda - regularization parameter for spatial TV for motion field
%   para  - struct containing additional parameters. See
%       OpticalFlowTVMotionEstimation_ADMM.m for most of them. In
%       addition:
%         'iter' iteration number to start from (for repeated calls by top
%                level function
%         'minEval' minimal number of operator evaluations before returning
%                   'energyOverShot' - see TVTVOF_Deblurring.m
%      in 'lsSolverPara':
%         'lsSolver' - 'CG', 'minres' (df), 'minresMat', 'AMG-CG':
%            minresMat and AMG-CG explicitly assemble the least-squares
%            matrix G'G
%          'lsMinIter' - min iterations for LS solver (df 1)
%          'preConFL' - boolean determining whether pre-conditioning
%            should be used for 'minresMat' (df: true)
%          'preConOptions' - struct with parameters for the
%            pre-conditioner, see ichol.m
%          'condEstAdaptationFL' - boolean determining whether the
%            conditon of the pre-conditioned system should be estimated to
%            balance the diagonal shift
%
%          'output' - boolean determining whether output should be displayed (df: true)
%          'outputPreStr' - pre-fix for output str (to structure output in multi-layered algos);
%          'outputFreq' - computational time in sec after which new output
%                          is displayed (to limit output in small-scale scenarios)
%          'returnAlgoVar'- boolean determining whether the auxillary variables of
%                           sub-routines should be returned to warm-start the sub-routines
%                           in the next call to TVTVOpticalFlowDenoising.m (df: true)
%          'rho' - scale of augmentation term (df: 10)
%          'rhoAdapt' - boolean controlling whether rho should be adapted (df: false)
%          'rhoAdaptMode' - way to adapt rho :'normal','slow','averaged','energy'
%                           (undocumented)
%          'mu'  - rho adaptation parameter, see Section 3.4.1 in Boyd et al, 2011
%          'tau' - rho adaptation parameter, see Section 3.4.1 in Boyd et al, 2011
%          'rhoMin' lower bound for rho
%          'computeEnergy' - boolean determining whether energy should be
%                              computed (df: true)
%          'computeIniEnergy' - boolean determining whether only the
%             energy of the initial guess should be computed and returned
%             (for compartibilty and precision reasons)
%
% OUTPUTS:
%       vBest - motion field with lowest energy found
%       v     - motion field in the last iteration
%       info  - struct which stores details and variables of the iteration,
%               can be used to inform subsequent calls to this function
%
% ABOUT:
% 	author          - Felix Lucka
% 	date            - 18.12.2018
% 	last update     - 27.10.2023
%
% see also ADMM.m, TV_FlowEstimation_ADMM.m


% =========================================================================
% CHECK INPUT AND INITILIZE VARIABLES
% =========================================================================


%%% get dimensions of problem
dim_f        = size(f1);
n            = prod(dim_f);
dim_space    = length(dim_f);
dim_v        = [dim_f, dim_space];
dim_z_jac    = [dim_v,dim_space];


%%% read out general algorithm parameters
% start iteration at this index (for consistent repeated runs)
iter             = checkSetInput(para, 'iter', 'i,>=0', 0);
max_eval         = checkSetInput(para, 'maxEval', 'i,>0', 1000);
min_eval         = checkSetInput(para, 'minEval', 'i,>0', 1);
max_iter         = ceil(max_eval/2);
stop_criterion   = checkSetInput(para, 'stopCriterion',...
                   {'primalDualRes','energyDec','maxEval'}, 'maxEval');
stop_tol         = checkSetInput(para, 'stopTolerance', '>=0', 10^-6);
typical_scale    = checkSetInput(para, 'typicalScale', '>=0', 1);
output           = checkSetInput(para, 'output', 'logical', false);
output_pre_str   = checkSetInput(para, 'outputPreStr', 'char', '');
output_freq      = checkSetInput(para, 'outputFreq', 'i,>0', 10);
return_algo_var  = checkSetInput(para, 'returnAlgoVar', 'logical', true);
energy_overshot  = checkSetInput(para, 'energyOverShot', '>0', 1.5);
data_cast        = checkSetInput(para, 'dataCast',...
                   {'single','double','gpuArray-single','gpuArray-double'}, 'double');
rho              = checkSetInput(para, 'rho', '>0', 10);
rho_daptation    = checkSetInput(para, 'rhoAdaptation', 'logical', false);
rho_adapt_mode   = checkSetInput(para, 'rhoAdaptMode', ...
                   {'normal','slow','averaged','energy'},'normal');
rho_adapt_K     = checkSetInput(para, 'rhoAdaptK', 'i,>0', Inf);
mu              = checkSetInput(para, 'mu', '>0', 10);
tau             = checkSetInput(para, 'tau', '>0', 2);
rho_min         = checkSetInput(para, 'rhoMin', '>0', 10^-10);
rho_max         = checkSetInput(para, 'rhoMax', '>0', 10^10);
damping_para    = checkSetInput(para, 'dampingPara', '>=0', 0);
overrelax_para  = checkSetInput(para, 'overRelaxPara', '>0', 1.8);
accelerate      = checkSetInput(para, 'accelerate','logical', false);
cmp_energy      = checkSetInput(para, 'computeEnergy', 'logical', true);
cmp_ini_energy  = checkSetInput(para, 'computeIniEnergy', 'logical', false);

info = [];

if( (~any(f1(:)) || ~any(f2(:))) && ~cmp_ini_energy)
    error('input u is all zero, return not implemented yet.')
end


%%% cast all variables to the desired type
[castFun, castZeros] = castFunction(data_cast);

cast_variables = {'a1','a2','f1','f2','v','z','w','rho','overrelax_para','lambda'};
if(output)
    disp([output_pre_str 'casting variables to type: ' data_cast])
end
for cast_index = 1:length(cast_variables)
    if(eval(['iscell(' cast_variables{cast_index} ')']))
        for iDim=1:length(cast_variables{cast_index})
            eval([cast_variables{cast_index} '{iDim} = castFun(' cast_variables{cast_index} '{iDim});']);
        end
    else
        eval([cast_variables{cast_index} ' = castFun(' cast_variables{cast_index} ');']);
    end
end


%%% read out least squares parameters
ls_solver_para = checkSetInput(para, 'lsSolverPara', 'struct', emptyStruct);
ls_solver     = checkSetInput(ls_solver_para,'lsSolver', ...
    {'CG','minres','minresMat','AMG-CG'}, 'minres');
ls_max_iter  = checkSetInput(ls_solver_para, 'maxIter', 'i,>0', min(max_iter,n));
ls_min_iter  = checkSetInput(ls_solver_para, 'minIter', 'i,>=0', 1);
ls_stop_criterion = checkSetInput(ls_solver_para, 'stopCriterion',...
    {'maxIter','relRes','progRelRes'}, 'relRes');
switch ls_stop_criterion
    case 'progRelRes'
        ls_prog_mode = checkSetInput(ls_solver_para, 'progMode', ...
            {'last','fac','poly'}, 'last');
        ls_tol       = checkSetInput(ls_solver_para, 'tol', '>0', 10^-4);
        switch ls_prog_mode
            case 'last'
                % nothing to be done here
            case 'fac'
                ls_tol_dec_fac = checkSetInput(ls_solver_para, 'tolDecFac', '>0', 0.99);
            case 'poly'
                ls_tol_dec_exp = checkSetInput(ls_solver_para, 'tolDecExp', '>0', 2);
                ls_tol_ini     = ls_tol;
                ls_tol         = ls_tol_ini / ((iter+1)^ls_tol_dec_exp);
        end
    case 'relRes'
        ls_tol = checkSetInput(ls_solver_para, 'tol', '>0', 10^-6);
    case 'maxIter'
        ls_tol = checkSetInput(ls_solver_para, 'tol', '>0', eps(castFun(1)));
end


%%% read out parameters defining the objective function and initial values
%%% for the split variables and lagrange multipliers
p        = checkSetInput(para, 'p', '>0', 2);
TV_type  = checkSetInput(para, 'TVtype',...
    {'anisotropic','mixedIsotropic','fullyIsotropic'}, 'mixedIsotropic');

switch p
    case 2
        split_OF = false;
    otherwise
        split_OF = true;
end

if(isempty(z))
    z_jac    = spatialFwdJacobian(v, false, true);
    if(split_OF)
       z_OF1     = castZeros(dim_f);
       z_OF_hat1 = z_OF1; % accelerated variable
       z_OF2     = castZeros(dim_f);
       z_OF_hat2 = z_OF2; % accelerated variable
    end
else
    if(split_OF)
       z_jac     = z{1};
       z_OF1     = z{2};
       z_OF_hat1 = z_OF1; % accelerated variable
       z_OF2     = z{3};
       z_OF_hat2 = z_OF2; % accelerated variable
    else
        z_jac = z;
    end
end
z_jac_hat = z_jac; % accelerated variable
clear z

if(isempty(w))
    w_jac = castZeros(dim_z_jac);
    if(split_OF)
        w_OF1     = castZeros(dim_f);
        w_OF_hat1 = w_OF1; % accelerated variable
        w_OF2     = castZeros(dim_f);
        w_OF_hat2 = w_OF2; % accelerated variable
    end
else
    if(split_OF)
       w_jac     = w{1};
       w_OF1     = w{2};
       w_OF_hat1 = w_OF1; % accelerated variable
       w_OF2     = w{3};
       w_OF_hat2 = w_OF2; % accelerated variable
    else
        w_jac = w;
    end
end
w_jac_hat = w_jac; % accelerated variable
clear w


%%% damping
if(damping_para > 0)
    [v_start, ~, para] = checkSetInput(para, 'vStart', 'numeric', v, [], true);
    v_start = castFun(v_start);
    damping_value = max(abs([a1(:);a2(:)])) * damping_para;
else
    damping_value = 0;
end


%%% initialize variables for the least squares solvers
switch ls_solver
    case 'CG'
        if(any(v(:)))
            %             [GGv,dfFL] = checkSetInput(ADMMVar,'GGv','numeric',1);
            %             if(dfFL)
            GGv = GGfun(v);
            %             else
            %                 GGv = castFun(GGv);
            %             end
        else
            GGv = v;
        end
    case 'minresMat'
        % set up matrices
        
        precon_flag = checkSetInput(ls_solver_para, 'preConFL', 'logical', true);
        precon_para = checkSetInput(ls_solver_para, 'preConOptions', 'struct', emptyStruct);
        if(precon_flag)
            precon_para.type     = checkSetInput(precon_para, 'type',...
                {'nofill','ict'}, 'nofill');
            precon_para.droptol  = checkSetInput(precon_para, 'droptol', '>0', 10^-2);
            precon_para.michol   = checkSetInput(precon_para, 'michol', ...
                {'on','off'}, 'off');
            precon_para.diagcomp = checkSetInput(precon_para, 'diagcomp',...
                '>=0', 0);
        end
        [GG_mat, i_chol_GG_mat] = assembleGGMat();
        
        cond_est_adaptation = checkSetInput(ls_solver_para, ...
            'condEstAdaptation', 'logical', false);
        if(cond_est_adaptation)
            cond_est_GG_mat = condestIChol(GG_mat, i_chol_GG_mat);
            cond_est_max   = checkSetInput(ls_solver_para, 'condEstMax', '>0', 10^6);
            while(cond_est_GG_mat > cond_est_max && rho_daptation)
                rho_old              = rho;
                % try larger rho
                rho                     = tau * rho_old;
                [GG_mat, i_chol_GG_mat] = assembleGGMat();
                cond_est_GG_mat_new     = condestIChol(GG_mat, i_chol_GG_mat);
                w_jac                   = w_jac / tau;
                if(split_OF)
                    w_OF1            = w_OF1 / tau;
                    w_OF2            = w_OF2 / tau;
                end
                if(cond_est_GG_mat_new > cond_est_GG_mat)
                    % try smaller rho
                    rho                 = rho_old / tau;
                    [GG_mat, i_chol_GG_mat] = assembleGGMat();
                    cond_est_GG_mat_new     = condestIChol(GG_mat, i_chol_GG_mat);
                    w_jac                = tau^2 * w_jac;
                    if(split_OF)
                        w_OF1            = tau^2 *  w_OF1;
                        w_OF2            = tau^2 *  w_OF2;
                    end
                    if(cond_est_GG_mat_new > cond_est_GG_mat)
                        tau = tau / 2;
                    else
                        cond_est_GG_mat = cond_est_GG_mat_new;
                    end
                else
                    cond_est_GG_mat = cond_est_GG_mat_new;
                end
            end
        end
    case 'AMG-CG'
        amg_opts             = [];
        amg_opts.tol         = ls_tol;
        amg_opts.solvermaxit = ls_max_iter;
        amg_opts.solverminit = ls_min_iter;
        amg_opts.setupflag   = false;
        amg_opts.printlevel  = 0;
        [GG_mat, AMG_var]    = assembleGGMat();
end


%%% initilize variables for energy monitoring and stop criteria
stop_value = Inf;
stop      = false;
cmp_primal_dual_res = checkSetInput(para, 'cmpPrimalDualRes', ...
    'logical', true);
switch stop_criterion
    case 'primalDualRes'
        cmp_primal_dual_res = true;
        eps_abs               = typical_scale * stop_tol;
        eps_rel               = stop_tol;
        stop_tol              = 1;
    case 'energyDec'
        cmp_energy = true;
end
cmp_primal_dual_res = rho_daptation || cmp_primal_dual_res;
n_eval       = 0;
eval_vs_iter = [];

if(cmp_energy)
    if(damping_para)
        v_start_dist    = sumAll((v - v_start).^2);
        info.vStartDist = castFun(checkSetInput(para, 'vStartDist', 'numeric', v_start_dist));
    end
    if(any(v(:)))
        OF_discrepancy = 1/p * sumAll(abs(sum(a1 .* v, dim_space+1) - f1).^p);
        OF_discrepancy = OF_discrepancy + 1/p * sumAll(abs(sum(a2 .* v, dim_space+1) - f2).^p);
        TVv    = TVofVelocityField(v, TV_type, true);
    else
        OF_discrepancy = 1/p * sumAll(abs(f1).^p);
        OF_discrepancy = OF_discrepancy + 1/p * sumAll(abs(f2).^p);
        TVv    = castZeros(1);
    end
    min_energy = castFun(checkSetInput(para, 'minEnergy', 'numeric', OF_discrepancy + lambda *  TVv));
    energy    = OF_discrepancy + lambda *  TVv;
    
    [v_best, ~, para] = checkSetInput(para, 'vBest', 'numeric', v, [], true);
    v_best            = castFun(v_best);
    
    info.OFDisc = castFun(checkSetInput(para, 'OFDisc', 'numeric', OF_discrepancy));
    info.TVv    = castFun(checkSetInput(para, 'TVv', 'numeric', TVv));
    info.iterBest = 0;
    if(cmp_ini_energy)
        return
    end
end

if(rho_daptation)
    rho_chain          = [];
    iter_without_adapt = 0;
end

if(cmp_primal_dual_res)
    primal_res_norm_hist =  [];
    dual_res_norm_hist   =  [];
end


%%% display output
if(output)
    output_str = output_pre_str;
    output_str = [output_str 'Starting OF + TV minimization with ADMM.'];
    if(cmp_energy)
        energy_here = energy(end);
        output_str = [output_str '; energy: ' num2str(energy_here, '%.4e')];
        last_output_energy = energy_here;
    end
    next_eval_output = output_freq;
    disp(output_str)
end

clear para


% =========================================================================
% MAIN ITERATION
% =========================================================================


while((n_eval < min_eval) || ((n_eval < max_eval) && ~stop))
    
    iter = iter + 1;
    
    
    %%% update v by solving least-squares problem
    switch ls_solver
        case 'CG'
            CGsolve()
        case 'minres'
            b_minres = setupRHS();
            G_minres = @(x) vec(GGfun(reshape(x, dim_v)));
            [v, ~, relres_ls, ls_iter, resvec_minres, resvec_cg_minres] = ...
                minres(G_minres, b_minres(:), ls_tol, ls_max_iter, [], [], v(:));
            if(ls_iter == 0)
                [v, ~, relres_ls, ls_iter, resvec_minres, resvec_cg_minres] = ...
                    minres(G_minres, b_minres(:), 0, 1, [], [], v(:));
            end
            n_eval = n_eval + 2 * ls_iter;
            v     = reshape(real(v), dim_v);
            clear *_minres
        case 'minresMat'
            b_minres = setupRHS();
            warning('off','MATLAB:minres:tooSmallTolerance');
            [v, flag_minres, relres_ls, ls_iter, resvec_minres, resvec_cg_minres] = ...
                minres(GG_mat, b_minres(:), ls_tol, ls_max_iter, i_chol_GG_mat, i_chol_GG_mat', v(:));
            if(ls_iter == 0) % do at least one step
                [v, flag_minres, relres_ls, ls_iter, resvec_minres, resvec_cg_minres] = ...
                    minres(GG_mat, b_minres(:), 0, 1, i_chol_GG_mat, i_chol_GG_mat', v(:));
            end
            warning('on','MATLAB:minres:tooSmallTolerance');
            
            switch flag_minres
                case 1
                    ls_iter = ls_max_iter;
            end
            n_eval = n_eval + 2 * ls_iter;
            v     = reshape(real(v), dim_v);
            clear *_minres
        case 'AMG-CG'
            b_minres      = setupRHS();
            amg_opts.x0   = v(:); clear v;
            [v, amgInfo] = amg(GG_mat,b_minres(:), amg_opts, AMG_var{1}, ...
                AMG_var{2}, AMG_var{3}, AMG_var{4}, AMG_var{5}, AMG_var{6});
            ls_iter       = amgInfo.itStep;
            relres_ls     = amgInfo.stopErr;
            n_eval        = n_eval + 2* ls_iter;
            v            = reshape(real(v),dim_v);
            clear *_minres
    end
    
    switch ls_stop_criterion
        case 'progRelRes'
            switch ls_prog_mode
                case 'fac'
                    ls_tol = ls_tol_dec_fac * ls_tol;
                case 'poly'
                    ls_tol = ls_tol_ini / ((iter+1)^ls_tol_dec_exp);
                case 'last'
                    ls_tol = relres_ls;
            end
            output_ls = ['itLS: ' int2str(ls_iter) ', rrLS: '  num2str(relres_ls,'%.2e')];
        case 'maxIter'
            output_ls = ['rrLS: ' num2str(relres_ls,'%.2e')];
        case 'relRes'
            output_ls = ['itLS: ' int2str(ls_iter)];
    end
    
    
    %%% update Ev
    Ev_jac            = spatialFwdJacobian(v, false, true);
    if(split_OF)
        Ev_OF1         = sum(a1 .* v, dim_space+1);
        Ev_OF2         = sum(a2 .* v, dim_space+1);
    end
    n_eval             = n_eval + 1;
    eval_vs_iter(end+1) = n_eval;
    
    if(cmp_energy)
        if(damping_para)
            v_start_dist = sumAll((v - v_start).^2);
        end
        
        OF_discrepancy = 1/p * sumAll(abs((sum(a1 .* v, dim_space+1) - f1)).^p) + 1/p * sumAll(abs((sum(a2 .* v, dim_space+1) - f2)).^p);
        
        switch TV_type
            case 'anisotropic'
                TVv = sumAll(abs(Ev_jac));
            case 'mixedIsotropic'
                TVv = sumAll(sqrt(sum(Ev_jac.^2, dim_space+2)));
            case 'fullyIsotropic'
                TVv = sumAll(sqrt(sum(sum(Ev_jac.^2, dim_space+2), dim_space+1)));
        end
        
        energy(end+1) = OF_discrepancy + lambda * TVv;
        if(energy(end) < min_energy)
            min_energy     = energy(end);
            v_best         = v;
            info.OFDisc   = gather(OF_discrepancy);
            info.TVv      = gather(TVv);
            info.iterBest = iter;
            if(damping_para > 0)
                info.vStartDist = gather(v_start_dist);
            end
        end
    end
    
    
    %%% over-relaxation
    aux_var_jac = overrelax_para * Ev_jac + (1-overrelax_para) * z_jac_hat; % F(z) = - z
    if(split_OF)
        aux_var_OF1 = overrelax_para * Ev_OF1 + (1-overrelax_para) * (z_OF_hat1 + f1); % E(u) + F(z) = b, F(z) = -z, b = -ut
        aux_var_OF2 = overrelax_para * Ev_OF2 + (1-overrelax_para) * (z_OF_hat2 + f2); % E(u) + F(z) = b, F(z) = -z, b = -ut
    end
    
    %%% update the split variable z
    z_jac_old = z_jac;
    if(split_OF)
        zOFOld1 = z_OF1;
        zOFOld2 = z_OF2;
    end
    
    switch TV_type
        case 'anisotropic'
            z_jac  = getfield(proxL1(aux_var_jac + w_jac_hat, lambda/rho), 'x');
        case 'mixedIsotropic'            
            z_jac = getfield(proxL21(aux_var_jac + w_jac_hat, lambda/rho, dim_space+2), 'x');
        case 'fullyIsotropic'
            notImpErr
    end
    if(split_OF)
        z_OF1 = getfield(proxLp(aux_var_OF1 - f1 + w_OF_hat1, 1/rho, p), 'x');
        z_OF2 = getfield(proxLp(aux_var_OF2 - f2 + w_OF_hat2, 1/rho, p), 'x');
    end
    
    
    %%% update the lagrange multipliers w
    w_jac_old = w_jac;
    if(split_OF)
        w_OF_old1 = w_OF1;
        w_OF_old2 = w_OF2;
    end
    
    w_jac    = w_jac_hat + (aux_var_jac - z_jac); % F(zGrad) = - zGrad
    if(split_OF)
        w_OF1 = w_OF_hat1 + (aux_var_OF1 - z_OF1 - f1); %
        w_OF2 = w_OF_hat2 + (aux_var_OF2 - z_OF2 - f2); %
    end
    
    if(cmp_primal_dual_res)
        % compute primal and dual residuum
        if(split_OF)
            primal_res_norm  = sqrt(sumAll((Ev_jac - z_jac).^2) + sumAll((Ev_OF1 - z_OF1 - f1).^2) + sumAll((Ev_OF2 - z_OF2 - f2).^2));
            dual_res_norm    = rho * sqrt(sumAll((spatialFwdJacobian(z_jac-z_jac_old, true, true) ...
                             + bsxfun(@times, a1, z_OF1-zOFOld1) + bsxfun(@times, a2, z_OF2-zOFOld2)).^2));
        else
            primal_res_norm  = sqrt(sumAll((Ev_jac - z_jac).^2));
            dual_res_norm    = rho * sqrt(sumAll((spatialFwdJacobian(z_jac-z_jac_old, true, true)).^2));
        end
        primal_res_norm_hist(end+1) = primal_res_norm;
        dual_res_norm_hist(end+1)   = dual_res_norm;
    end
    clear Ez* auxVar*
    
    %%% acceleration 
    if(accelerate)
        gamma = iter/(iter + 3);
        z_jac_hat  = z_jac + gamma * (z_jac - z_jac_old);  
        w_jac_hat  = w_jac + gamma * (w_jac - w_jac_old);  
        if(split_OF)
            w_OF_hat1  = w_OF1 + gamma * (w_OF1 - w_OF_old1); 
            w_OF_hat2  = w_OF2 + gamma * (w_OF2 - w_OF_old2); 
            z_OF_hat1  = z_OF1 + gamma * (z_OF1 - zOFOld1); 
            z_OF_hat2  = z_OF2 + gamma * (z_OF2 - zOFOld2); 
        end
    else
        z_jac_hat  = z_jac;
        w_jac_hat  = w_jac;
        if(split_OF)
            z_OF_hat1 = z_OF1;
            z_OF_hat2 = z_OF2;
            w_OF_hat1  = w_OF1;
            w_OF_hat2  = w_OF2;
        end
    end
    clear z_jac_old w_jac_old zOFOld1 w_OF_old1 zOFOld2 w_OF_old2
    
    %%% plotting and output
    if(output && n_eval >= next_eval_output)
        output_str =  [output_pre_str 'it ' int2str(iter)];
        output_str =  [output_str ', evl ' int2str(n_eval) '/' int2str(max_eval)];
        output_str = [output_str  '; ' output_ls];
        
        if(cmp_primal_dual_res)
            output_str = [output_str  '; pr/du res: '  ...
                num2str(primal_res_norm, '%.2e') '/' num2str(dual_res_norm, '%.2e')];
        end
        if(rho_daptation)
            output_str = [output_str  '; rho: '  num2str(rho, '%.2e')];
        end
        
        if(cmp_energy)
            energy_here = energy(end);
            output_str = [output_str '; en: ' num2str(energy_here, '%.2e') ...
                ' (dFac: ' num2str((last_output_energy-energy_here)/last_output_energy, '%.2e') ')'];
            if(all(info.iterBest < iter))
                output_str = [output_str '*'];
            end
            last_output_energy = energy_here;
        end
        disp(output_str)
        next_eval_output = n_eval + output_freq;
    end
    
    
    %%% update rho
    if(rho_daptation)
        rho_old = rho;
        switch rho_adapt_mode
            case {'normal'}
                prim_val  = primal_res_norm;
                dual_val  = dual_res_norm;
                adapt_fac = tau;
            case 'slow'
                adapt_fac = (tau + iter_without_adapt) / (iter_without_adapt + 1);
            case 'averaged'
                non_zeros = primal_res_norm_hist > 0;
                if(isempty(non_zeros))
                    prim_val = 0;
                else
                    prim_val = exp(mean(log(primal_res_norm_hist(non_zeros))));
                end
                non_zeros = dual_res_norm_hist > 0;
                if(isempty(non_zeros))
                    dual_val = 0;
                else
                    dual_val = exp(mean(log(dual_res_norm_hist(non_zeros))));
                end
                adapt_fac = tau;
            case 'energy'
                dual_val  = 1;
                prim_val  = 1;
                adapt_fac = tau;
                if(energy(end) > energy(end-1))
                    prim_val = mu + 1;
                elseif(dual_res_norm > mu * primal_res_norm)
                    % normal adaptation
                    prim_val  = primal_res_norm;
                    dual_val  = dual_res_norm;
                    adapt_fac = tau;
                end
        end
        
        if(prim_val > mu * dual_val && rho*adapt_fac < rho_max)
            rho = adapt_fac * rho;
            if(isvarname('GGMat'))
                switch ls_solver
                    case 'CG'
                        [GG_mat, i_chol_GG_mat] = assembleGGMat();
                        if(cond_est_adaptation)
                            cond_est_GG_mat = condestIChol(GG_mat, i_chol_GG_mat);
                            if(cond_est_GG_mat > cond_est_max)
                                adapt_fac = 1;
                                rho      = rho_old;
                                rho_daptation = false;
                                [GG_mat, i_chol_GG_mat] = assembleGGMat();
                                cond_est_GG_mat = condestIChol(GG_mat, i_chol_GG_mat);
                            end
                        end
                    case 'AMG-CG'
                        [GG_mat, AMG_var] = assembleGGMat();
                end
            elseif(isvarname('GGv'))
                GGv = GGfun(v);
            end
            
            w_jac     = w_jac    / adapt_fac;
            w_jac_hat  = w_jac_hat / adapt_fac;
            if(split_OF)
                 w_OF1    = w_OF1 / adapt_fac;
                 w_OF2    = w_OF2 / adapt_fac;
                 w_OF_hat1 = w_OF_hat1 / adapt_fac;
                 w_OF_hat2 = w_OF_hat2 / adapt_fac;
            end
            n_eval = n_eval + 2;
            iter_without_adapt = 0;
        elseif(dual_val > mu * prim_val && rho/adapt_fac > rho_min)
            rho = rho / adapt_fac;
            if(isvarname('GGMat'))
                switch ls_solver
                    case 'CG'
                        [GG_mat, i_chol_GG_mat] = assembleGGMat();
                        if(cond_est_adaptation)
                            cond_est_GG_mat = condestIChol(GG_mat, i_chol_GG_mat);
                            if(cond_est_GG_mat > cond_est_max)
                                adapt_fac = 1;
                                rho      = rho_old;
                                rho_daptation = false;
                                [GG_mat, i_chol_GG_mat] = assembleGGMat();
                                cond_est_GG_mat = condestIChol(GG_mat, i_chol_GG_mat);
                            end
                        end
                    case 'AMG-CG'
                        [GG_mat, AMG_var] = assembleGGMat();
                end
            elseif(isvarname('GGv'))
                GGv = GGfun(v);
            end
            
            w_jac        = adapt_fac * w_jac;
            w_jac_hat     = adapt_fac * w_jac_hat;
            if(split_OF)
               w_OF1      = adapt_fac * w_OF1;
               w_OF2      = adapt_fac * w_OF2;
               w_OF_hat1   = adapt_fac * w_OF_hat1;
               w_OF_hat2   = adapt_fac * w_OF_hat2;
            end
            n_eval = n_eval + 2;
            iter_without_adapt = 0;
        else
            iter_without_adapt = iter_without_adapt + 1;
        end
        rho_chain(end+1) = rho;
        
        if(iter > rho_adapt_K)
            rho_daptation = false;
        end
    end
    
    
    %%% check stop conditions
    switch stop_criterion
        case 'energyDec'
            if(~stop && min_energy < energy(1))
                first_energy_dec = find(energy(2:end) <= energy(1),1,'first') + 1;
                if(~isempty(first_energy_dec)  && first_energy_dec <= ceil(length(energy)/energy_overshot))
                    stop = true;
                end
            end
        case 'maxEval'
            
        otherwise
            notImpErr
    end
    
    
end


% =========================================================================
% GATHER RESULTS AND CLEAN UP
% =========================================================================


%%% free memory
switch ls_solver
    case 'minresMat'
        clear GG_mat i_chol_GG_mat
    case 'AMG-CG'
        clear GG_mat AMG_var
end

%%% gather all variables
v_best = gather(v_best);

if(return_algo_var)
    if(split_OF)
        z{1} = gather(z_jac); clear z_jac
        z{2} = gather(z_OF1); clear z_OF1
        z{3} = gather(z_OF2); clear zO2F
        w{1} = gather(w_jac); clear w_jac
        w{2} = gather(w_OF1); clear w_OF1
        w{3} = gather(w_OF2); clear wO2F
    else
        z = gather(z_jac); clear z_jac
        w = gather(w_jac); clear w_jac
    end
else
    z = [];
    w = [];
end

info.nEval         = n_eval;
info.iter          = iter;
info.rho           = gather(rho);
info.rhoAdaptation = rho_daptation;
info.evalVsIter    = eval_vs_iter;

if(cmp_energy)
    info.energy     = gather(energy);
    info.minEnergy  = gather(min_energy);
end

if(exist('rhoChain','var'))
    info.rhoChain = gather(rho_chain);
end

if(cmp_primal_dual_res)
    info.primalResNorm = gather(primal_res_norm_hist);
    info.dualResNorm   = gather(dual_res_norm_hist);
end

info.lsSolverPara = emptyStruct;
switch ls_stop_criterion
    case 'progRelRes'
        switch ls_prog_mode
            case {'last','fac'}
                info.lsSolverPara.tol = ls_tol;
        end
end

%     function res = GGfun(vIn,lambdaHere)
%         res = bsxfun(@times,uSpaCenGrad,sum(uSpaCenGrad.*vIn,dimSpace+2))/rho;
%         res = res + lambdaHere^2 * spatialFwdJacobian(spatialFwdJacobian(vIn,false),true);
%         if(dampingPara > 0)
%             res = res + dampingValue^2 * vIn;
%         end
%     end


%     function res = Gfun(x,lambdaHere,adjointStr)
%         switch adjointStr
%             case 'notransp'
%                 res = [vec(sum(uSpaCenGrad.*x,dimSpace+1))/sqrt(rho);vec(lambdaHere * (spatialFwdJacobian(x,false,true)))];
%                 if(dampingPara > 0)
%                     res = [res;dampingValue * vec(x)];
%                 end
%             case 'transp'
%                 res   = bsxfun(@times,uSpaCenGrad,reshape(x(1:n),dimU))/sqrt(rho);
%                 res   = res + lambdaHere * spatialFwdJacobian(reshape(x(n+1:((dimSpace^2+1)*n)),dimZ),true,true);
%                 if(dampingPara > 0)
%                     res = res + dampingValue/sqrt(rho) * reshape(x(((dimSpace^2+1)+1):end),dimV);
%                 end
%                 res = res(:);
%             otherwise
%                 error('adjointFL has to be ''transp'' or ''notransp''')
%         end
%     end


% =========================================================================
% NESTED FUNCTIONS
% =========================================================================


    %%% compute G' G * v for the linear operator G in the least-squares problem
    function res = GGfun(vIn)
        res = zeros(dim_v, 'like', vIn);
        switch dim_space
            case 2
                % partial derivatives in x dir
                res(1,:,:)       = vIn(1,:,:,:) - vIn(2,:,:);
                res(2:end-1,:,:) = -diff(vIn,2,1);
                res(end,:,:)     = vIn(end,:,:) - vIn(end-1,:,:);
                % partial derivatives in y dir
                res(:,1,:)       = res(:,1,:) + (vIn(:,1,:) - vIn(:,2,:));
                res(:,2:end-1,:) = res(:,2:end-1,:) - diff(vIn,2,2);
                res(:,end,:)     = res(:,end,:) + (vIn(:,end,:) - vIn(:,end-1,:));
            case 3
                % partial derivatives in x dir
                res(1,:,:,:)       = vIn(1,:,:,:) - vIn(2,:,:,:);
                res(2:end-1,:,:,:) = -diff(vIn,2,1);
                res(end,:,:,:)     = vIn(end,:,:,:) - vIn(end-1,:,:,:);
                % partial derivatives in y dir
                res(:,1,:,:)       = res(:,1,:,:)+ (vIn(:,1,:,:) - vIn(:,2,:,:));
                res(:,2:end-1,:,:) = res(:,2:end-1,:,:) - diff(vIn,2,2);
                res(:,end,:,:)     = res(:,end,:,:) + (vIn(:,end,:,:) - vIn(:,end-1,:,:));
                % partial derivatives in z dir
                res(:,:,1,:)       = res(:,:,1,:)+ (vIn(:,:,1,:) - vIn(:,:,2,:));
                res(:,:,2:end-1,:) = res(:,:,2:end-1,:) - diff(vIn,2,3);
                res(:,:,end,:)     = res(:,:,end,:) + (vIn(:,:,end,:) - vIn(:,:,end-1,:));
            otherwise
                notImpErr
        end
        
        if(split_OF)
            res = res + bsxfun(@times, a1, sum(a1 .* vIn, dim_space+1));
            res = res + bsxfun(@times, a2, sum(a2 .* vIn, dim_space+1));
            if(damping_para > 0)
                res = res + damping_value^2 * vIn;
            end
        else
            res = res + bsxfun(@times, a1, sum(a1 .* vIn, dim_space+1)) / rho;
            res = res + bsxfun(@times, a2, sum(a2 .* vIn, dim_space+1)) / rho;
            if(damping_para > 0)
                res = res + damping_value^2/rho * vIn;
            end
        end

    end


    %%% assemble G'*G as a matrix and compute pre-conditioner
    function [GG_mat, dec_GG_mat] = assembleGGMat()
        
        % assemble G'*G
        GG_mat = castFun(laplaceMatrix(dim_f, 'NB'));
        eval(['GG_mat = blkdiag(GG_mat' repmat(',GG_mat', [1, dim_space-1]) ');']);
        for i=1:dim_space
            for j=1:dim_space
                switch dim_space
                    case 2
                        trans_part{i,j} =  spdiags(vec(a1(:,:,i) .* a1(:,:,j)), 0, n, n)...
                                        + spdiags(vec(a2(:,:,i) .* a2(:,:,j)), 0, n, n) ;
                    case 3
                        trans_part{i,j} =  spdiags(vec(a1(:,:,:,i) .* a1(:,:,:,j)), 0, n, n)...
                                        + spdiags(vec(a2(:,:,:,i) .* a2(:,:,:,j)), 0, n, n);
                end
            end
        end
        if(split_OF)
            GG_mat = GG_mat + cell2mat(trans_part);
        else
            GG_mat = GG_mat + cell2mat(trans_part) / rho;
        end
        
        clear trans_part
        
        % compute a pre-conditioner
        dec_GG_mat = [];
        switch ls_solver
            case 'minresMat'
                if(precon_flag)
                    while(1)
                        % use ichol but make sure it is positive definite
                        try
                            dec_GG_mat = ichol(GG_mat, precon_para);
                            break
                        catch ME
                            switch ME.message
                                case 'Encountered nonpositive pivot.'
                                    precon_para.diagcomp = max(10^-10, 13*precon_para.diagcomp);
                                otherwise
                                    rethrow(ME)
                            end
                        end
                    end
                    precon_para.diagcomp = precon_para.diagcomp/11;
                end
            case 'AMG-CG'
                % use algebraic multi-grid pre-conditioner
                dec_GG_mat            = {};
                amg_opt_setup        = [];
                amg_opt_setup.solver = 'NO';
                [~, ~, dec_GG_mat{1}, dec_GG_mat{2}, dec_GG_mat{3}, dec_GG_mat{4}, dec_GG_mat{5}, dec_GG_mat{6}] = ...
                    amg(GG_mat, ones(size(GG_mat, 1), 1), amg_opt_setup);
        end
    end

    
    %%% use conjugate gradient to solve least-squares system
    function CGsolve()
        %%% update u by solving G'G u = G'c via CG
        % adjoint of sum_i uSpaGrad_i * v_i
        r_cg     = spatialFwdJacobian(z_jac_hat - w_jac_hat, true, true);
        if(split_OF)
            r_cg   = r_cg + bsxfun(@times, a1, z_OF_hat1 + f1 - w_OF_hat1);
            r_cg   = r_cg + bsxfun(@times, a2, z_OF_hat2 + f2 - w_OF_hat2);
            if(damping_para > 0)
                r_cg = r_cg + damping_value^2 * v_start;
            end
        else
            r_cg   = r_cg + bsxfun(@times, a1, + f1/rho);
            r_cg   = r_cg + bsxfun(@times, a2, + f2/rho);
            if(damping_para > 0)
                r_cg = r_cg + damping_value^2/rho * v_start;
            end
        end
        n_eval = n_eval + 1;
        norm_rhs = sqrt(sum(r_cg(:).^2));
        r_cg     = r_cg - GGv;
        
        rho_cg          = sumAll(r_cg.*r_cg);
        min_relres_cg   = sqrt(rho_cg) / norm_rhs;
        v_min_relres_cg = v;
        p_cg            = r_cg;
        
        for i_cg = 1:ls_max_iter
            q_cg        = GGfun(p_cg, 1);
            alpha_cg    = rho_cg / sumAll(p_cg .* q_cg);
            v           = v   + alpha_cg * p_cg;
            GGv         = GGv + alpha_cg * q_cg;
            r_cg        = r_cg - alpha_cg * q_cg;
            rho_new_cg  = sumAll(r_cg .* r_cg);
            relres_cg   = sqrt(rho_new_cg) / norm_rhs;
            if(relres_cg < min_relres_cg)
                v_min_relres_cg = v;
                min_relres_cg  = relres_cg;
            end
            if(relres_cg < ls_tol && ~strcmp(ls_stop_criterion, 'maxIter'))
                break
            end
            p_cg      = r_cg + (rho_new_cg / rho_cg) * p_cg;
            rho_cg    = rho_new_cg;
        end
        
        v                 = v_min_relres_cg;
        info.lsIter(iter) = i_cg;
        n_eval             = n_eval + 2*i_cg;
        relres_ls          = relres_cg;
        clear *CG
    end


     %%% set up right-hand-side b of least squares system G c = b
    function b_minres = setupRHS()
        
        b_minres = spatialFwdJacobian(z_jac_hat - w_jac_hat, true, true);
        if(split_OF)
            b_minres = b_minres + bsxfun(@times, a1, z_OF_hat1 + f1 - w_OF_hat1);
            b_minres = b_minres + bsxfun(@times, a2, z_OF_hat2 + f2 - w_OF_hat2);
            if(damping_para > 0)
                b_minres = b_minres + damping_value^2 * v_start;
            end
        else
            b_minres = b_minres + bsxfun(@times, a1, + f1/rho);
            b_minres = b_minres + bsxfun(@times, a2, + f2/rho);
            if(damping_para > 0)
                b_minres = b_minres + damping_value^2/rho * v_start;
            end
        end
        n_eval = n_eval + 1;

    end


end
