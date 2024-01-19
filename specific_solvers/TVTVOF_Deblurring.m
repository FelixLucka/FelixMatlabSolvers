function [u_return, v_return, info]  =  TVTVOF_Deblurring(A, AT, f_or_ATf, alpha,...
    beta, gamma, para)
%TVTVOF_DEBLURRING performs joint TV deblurring and TV-regularized motion
% estimation based on optical flow models
%
% DESCRIPTION:
% 	TVTVOF_Deblurring takes a temporal sequence of data and
%   performs joint image reconstruction and motion estimation. The broad idea
%   is described in
%   "Enhancing Compressed Sensing Photoacoustic Tomography by Simultaneous
%   Motion Estimation" by Arridge, Beard, Betcke, Cox, Huynh, Lucka Zhang,
%   2018.
%   but the algorithm here does not split of the forward operator A off.
%   The large scale non-linear optical flow part follows
%   "Joint Large Scale Motion Estmation and Image Reconstruction" by
%   Hendrink Dirks, 2018.
%
%   The algorithm tries to minimize the following energy:
%   (u,v) = argmin { \sum_t^T  1/2 \| A_t u_t - f_t \|_2^2 + \alpha_t \|\nabla u_t \|_1
%                + \beta_t \sum_i^d \|\nabla v_{x_i,t} \|_1
%                + \gamma_t^p/p \| rho(u_{t+1}, u_t, v_t) \|_p^p }
%   OR if symmetricOF = true
%   (u,v) = argmin { \sum_t^T  1/2 \| A_t u_t - f_t \|_2^2 + \alpha_t \|\nabla u_t \|_1
%                + \beta_t \sum_i^d \|\nabla v_{x_i,t} \|_1
%                + \gamma_t^p/(2*p) \| rho(u_{t+1}, u_t, v_t) \|_p^p
%                + \gamma_t^p/(2*p) \| rho(u_{t}, u_{t+1}, -v_t) \|_p^p}
%
%   where rho(u_{t+1}, u_t, v_t) = u_{t+1} - u_t + (\nabla u_{t+1}) \cdot v_t
%      or rho(u_{t+1}, u_t, v_t) = warp(u_{t+1} v_t) - u_t
%   The algorithm alternates minimization over u and v. Each sub-problem in u and v can
%   be solved via PDGH or ADMM. For a general reference, we refer to
%   "An introduction to continuous optimization for imaging" by Chambolle and
%   Pock, 2016.
%
% USAGE:
%       [u,v,info]  =  TVTVOF_Deblurring(A, AT, f,alpha, beta, gamma, para)
%
% INPUTS:
%   A      - function handle to the forward operator
%   AT     - function handle to the adjoint of the forward operator
%   f_or_ATf - either the data f as cell array or AT(f) as 2D+1D or
%            3D+1D numerical array (needs to be specified in para)
%   alpha - regularization parameter for spatial TV for u (can be vector of
%           length T)
%   beta  - regularization parameter for spatial TV for v (can be vector of
%           length T-1)
%   gamma - regularization parameter for motion coupling (can be vector of
%           length T-1)
%   para  - struct containing additional parameters:
%       'useATF' -  logical indicating whether fOrATf contains f (false)
%                   or AT*f (true)
%        'dimU'  - dimension of u as [nX, nX, T] or [nX, nX, nZ, T]
%
%     energy function specification:
%       'OFtype' - 'linear' (default) or 'non-linear'
%       'p'      - p of the Lp norm to inforce optical flow constraint:
%                  1 <= p <= 2 (default p = 2)
%       'symmetricOF' - boolean determining whether the OF part is
%       symmetric in u_1 and u_2 by adding two OF terms (see above, default: false)
%       'TVtypeU' - total variation type used for image
%                   'isotropic' (df) or 'anisotropic'
%       'TVtypeV' - total variation type used for motion field
%                   'anisotropic', 'mixedIsotropic' (df), 'fullyIsotropic'
%       'constraint' - constraint on image: 'none' (df), 'non-negative' 'range'
%
%     general parameters of the alternation:
%
%       'u' - initial value for u
%       'v' - initial value for v
%       'stopCriterion':
%         'maxIter' - for stopping after maximal number of iterations (df)
%         'relEnergy' - for stopping after the relative decrease of energy
%                       between two iteration is below a certain tolerance
%         'compTime' - after a certain computational time is spend
%         'stopTolerance' - tolerance for stopCriterion 'relEnergy'
%         'maxIter' - maximal number of alternations
%         'minIter' - minimal number of alternations
%         'maxCompTime' - maximal computational time the algorithm is allowed to spend
%         'hardCompRestrFL' - boolean determining whether computation time
%                             constraint should be enforced in a hard way,
%                             i.e., ignoring other constraints (df: false)
%         'monotoneEnergyFL' - boolean determining whether a monoton energy decrease
%                              in the sub-functions should be enforced (df = true)
%         'energyOverShot' - if > 1, sub-algorithms will run longer after achieving
%                            an energy decrease, for energyOverShot = a they will run
%                            for a total of a*iterDec iterations where iterDec
%                            is the number of iterations after which an
%                            energy decrease was achieved for the first
%                            time
%
%       'overrelaxPara' - if > 1, overrelaxatin will be performed (df = 1)
%       'inertiaPara' - if > 0, inertia will be added to the (u,v) move
%                       (see Chambolle & Pock 2016)
%       'addLineSearchFL' - boolean determining whether a line-search is added
%                            from (uOld,vOld) to the updated (u,v) (df = false)
%       'outputFL' - boolean determining whether output should be displayed (df: true)
%       'outputPreStr' - pre-fix for output str (to structure output in multi-layered algos);
%       'outputFreq' - computational time in sec after which new output
%                      is displayed (to limit output in small-scale scenarios)
%       'returnOptVar'- boolean determining whether the auxillary variables of
%                       sub-routines should be returned to warm-start the sub-routines
%                       in the next call to TVTVOpticalFlowDenoising.m (df: false)
%
%     parameters specifing which algorithms to use in the sub steps:
%       'uUpdateAlgo' - 'PDHG' (df) ,'ADMM' determines whether
%             TVplusOptical%   fOrATf - either the data f as cell array or AT(f) as 2D+1D or
%            3D+1D numerical array (needs to be specified in para)FlowDenosing_PDHG.m or
%             TVplusOpticalFlowDenosing_ADMM will be called for u-update
%       'uOptPara' - struct with parameters for the u-update:
%             'warmstartUFL' - warmstart algorithm from previous iteration (df true)
%             see documentation of TVplusOpticalFlowDenosing_* for other fields
%       'vUpdateAlgo' - 'PDHG' (df) ,'ADMM' determines whether
%             OpticalFlowTVMotionEstimation_PDHG.m or
%             OpticalFlowTVMotionEstimation_ADMM will be called for v-update
%       'vOptPara' - struct with parameters for the v-update:
%       'warmstartVFL' - warmstart algorithm from previous iteration (df true)
%       'vMode' - 'block' or 'FbF', depending on whether the
%                  computation of the v-update should be computed in a block or for
%                  each frame separetly
%       'parallelVFL' - boolean determining whether the single frames
%              for 'vMode' = 'FbF', should be computed with parallel pool
%       'nWorkerPool' - if the above is true, the number of workers in the
%               matlab pool that are used; see documentation of
%               OpticalFlowTVMotionEstimation_* for other fields
%
%
% OUTPUTS:
%   (u_return, v_return) - (u,v) pair which gave the lowest energy
%   info - struct containing additional information and variables of
%   the iteration, can be used to inform subsequent calls to this
%   function
%
%
% ABOUT:
%   based on code by Hendrik Dirks,
%   https://github.com/HendrikMuenster/JointMotionEstimationAndImageReconstruction
% 
%   author          - Felix Lucka
%   date            - 06.05.2018
%   last update     - 21.12.2023
%
% See also

clock_cmp_total = tic; % track total computation time

% =========================================================================
% CHECK INPUT AND INITILIZE VARIABLES
% =========================================================================

info = [];

%%% read out dimensions
use_ATf        = checkSetInput(para, 'useATF', 'logical', false);
if(use_ATf)
    sqnorm_f = checkSetInput(para, 'sqNormF', '>=0', 0);
    sz_u     = size(f_or_ATf);
else
    % we're given f we need to get the size of U through parameters or by
    % one application of A' * f
    [sz_u, df_v] = checkSetInput(para, 'szU', 'i,>0', 0);
    if(df_v)
        sz_u = size(AT(f_or_ATf));
    end
    if(iscell(f_or_ATf))
        sqnorm_f = sum(cellfun(@(x) sum(x(:).^2), f_or_ATf));
    else
        sqnorm_f = sum(f_or_ATf(:).^2);
    end
end
dim_space   = length(sz_u(1:end-1));
n_xyz       = sz_u(1:dim_space);
n_t         = sz_u(end);
sz_v        = [n_xyz, dim_space, n_t-1];


%%% read out parameters defining the energy function
OF_type      = checkSetInput(para, 'OFtype', {'linear', 'nonLinear'}, 'linear');
p            = checkSetInput(para, 'p', '>=1', 2);
symmetric_OF = checkSetInput(para, 'symmetricOF', 'logical', false);
TV_type_u    = checkSetInput(para, 'TVtypeU', {'isotropic','anisotropic'}, ...
    'isotropic');
TV_type_v    = checkSetInput(para, 'TVtypeV', {'anisotropic','mixedIsotropic',...
    'fullyIsotropic'},'mixedIsotropic');
constraint    = checkSetInput(para, 'constraint', {'none','nonNegative','range'},...
    'none');
warp_para        = checkSetInput(para, 'warpPara', 'struct', emptyStruct);
OF_para          = [];
OF_para.warpPara = warp_para;
OF_para.dimSpace = dim_space;

%%% read out parameters of the bi-convex search algorithm (outer loop)
stop_criterion = checkSetInput(para, 'stopCriterion', ...
    {'maxIter','relEnergy','compTime'}, 'maxIter');
switch stop_criterion
    case 'maxIter'
        df_max_iter     = 'error';
        df_max_cmp_time = Inf;
    case 'relEnergy'
        df_max_iter     = Inf;
        df_max_cmp_time = Inf;
    case 'compTime'
        df_max_iter     = Inf;
        df_max_cmp_time = 'error';
end
stop_tol = checkSetInput(para, 'stopTolerance', '>=0', 10^-6);
max_iter       = checkSetInput(para, 'maxIter', 'i,>0', df_max_iter);
min_iter       = checkSetInput(para, 'minIter', 'i,>=0', 0);
max_cmp_time   = checkSetInput(para, 'maxCompTime', '>0', df_max_cmp_time);
hard_cmp_restr = checkSetInput(para, 'hardCompRestr', 'logical', false);
% ensure monotone energy decrease of alternating scheme?
monotone_energy = checkSetInput(para,'monotoneEnergy', 'logical', true);
if(monotone_energy)
    % allowedMaxEvalAdjust = {'none','compTime','energyDec','half'};
    % continue sub-routines until sufficient energy decrease is reached?
    energy_overshot = checkSetInput(para, 'energyOverShot', '>=1', 2);
else
    % allowedMaxEvalAdjust = {'none','compTime','half'};
end
% % how to adjust the operator evaluations in the sub routines
%adjustMaxEval   = checkSetInput(para,'adjustMaxEval',allowedMaxEvalAdjust,'none');
% use over relaxation, inertia or linesearch?
overrelax_para  = checkSetInput(para, 'overrelaxPara', '>=0', 1);
inertia_para    = checkSetInput(para, 'inertiaPara', '>=0', 0);
line_search     = checkSetInput(para, 'lineSearch', 'logical', false);
% output settings2
output          = checkSetInput(para, 'output', 'logical', true);
output_pre_str  = checkSetInput(para, 'outputPreStr', 'char', '');
output_freq     = checkSetInput(para, 'outputFreq', '>0', 0);
% return optimization variables?
return_opt_var  = checkSetInput(para, 'returnOptVar', 'logical', false);


%%% set up energy functions
if(isscalar(alpha))
    alpha = alpha * ones(n_t, 1);
end
if(isscalar(beta))
    beta = beta * ones(n_t-1, 1);
end
if(isscalar(gamma))
    gamma = gamma * ones(n_t-1, 1);
end

% ratio of beta and gamma will be used in the motion estimation part
lambda  = beta./(gamma.^p);

% data fitting term
if(use_ATf)
    dataFitFunAux = @(Au, u) sumAll(Au .* Au) - 2 * sumAll(u .* f_or_ATf) + sqnorm_f;
    dataFitFun    = @(u) 0.5 * dataFitFunAux(double(A(u)), double(u));
else
    if(iscell(f_or_ATf))
        dataFitFun = @(u) 0.5 * sum(cellfun(@(x,y) sumAll((x-y).^2), A(u), f_or_ATf));
    else
        dataFitFun = @(u) 0.5 * sum(vec(A(u) - f_or_ATf).^2);
    end
end
% TV regularization on u
switch TV_type_u
    case 'anisotropic'
        TVuFun = @(u) sumAllBut(abs(spatialFwdGrad(u, false)), 'last');
    case 'isotropic'
        TVuFun = @(u) sumAllBut(sqrt(sum((spatialFwdGrad(u, false)).^2, dim_space+1)), 'last');
end
% TV regularization on v
switch TV_type_v
    case 'anisotropic'
        TVvFun = @(v) sumAllBut(abs(spatialFwdJacobian(v, false)), 'last');
    case 'mixedIsotropic'
        TVvFun = @(v) sumAllBut(sqrt(sum((spatialFwdJacobian(v, false)).^2, dim_space+2)), 'last');
    case 'fullyIsotropic'
        TVvFun = @(v) sumAllBut(sqrt(sum(sum((spatialFwdJacobian(v,false)).^2,...
            dim_space+2), dim_space+1)), 'last');
end
% Optical flow type
if(symmetric_OF)
    OFfun = @(u,v) 1/(2*p) * sumAllBut(abs(opticalFlowOperator(u, v, OF_type, false, OF_para, false)).^p, 'last') ...
                 + 1/(2*p) * sumAllBut(abs(opticalFlowOperator(u, v, OF_type, false, OF_para, true )).^p, 'last');
else
    OFfun = @(u,v) 1/p *sumAllBut(abs(opticalFlowOperator(u, v, OF_type, false, OF_para, false)).^p, 'last');
end
% box constraints on u
switch constraint
    case {'none', 'nonNegative'}
        con_range = [];
    case 'range'
        con_range = checkSetInput(para, 'conRange', 'double', [], 'error');
        if(~ length(con_range) == 2 && con_range(1) < con_range(2))
            error(['invalid range: ' num2str(con_range)])
        end
end
conFunc      = @(u) energyBoxConstraints(u, constraint, con_range);
projectionU  = @(u) getfield(projBoxConstraints(u, constraint, con_range), 'x');

% complete energy functions
energyFunWithoutCon = @(u,v) dataFitFun(u) + sum(alpha .* TVuFun(u)) ...
    + sum(beta .* TVvFun(v)) + sum(gamma.^p .* OFfun(u, v));
energyFun           = @(u,v) energyFunWithoutCon(u, v) + conFunc(u);


%%% read out parameters specifing which algorithms to use in the sub steps
% u update
[u_opt, ~, para] = checkSetInput(para, 'uOpt', 'struct', emptyStruct, [], true);
% v update
[v_opt, ~, para] = checkSetInput(para, 'vOpt', 'struct', emptyStruct, [], true);


%%% initial values for u and v
[u, df_u, para] = checkSetInput(para, 'u', 'numeric', zeros(sz_u), [], true);
u               = projectionU(u);
[v, df_v, para] = checkSetInput(para, 'v', 'numeric', zeros(sz_v), [], true);

if(df_u)
    u_ini_mode = checkSetInput(para, 'uIniMode', {'normal', 'TV-FbF', 'TVOF'}, 'normal');
else
    u_ini_mode = 'skip';
end



%%% parameter settings for the for the u update algorithm
warmstart_u  = checkSetInput(u_opt, 'warmstart', 'logical', true);
u_solver      = checkSetInput(u_opt, 'solver', {'PDHG', 'ADMM'}, 'ADMM');
% overwrite some defaults for the solver
u_opt.algo    = checkSetInput(u_opt, 'algo', 'struct', emptyStruct, [], true);
% copy settings for the energy function into solver para
u_opt.algo = mergeStructs(u_opt.algo, struct('useATF', use_ATf, 'TVtype', TV_type_u, ...
    'OFtype', OF_type, 'p', p, 'symmetricOF', symmetric_OF, 'constraint', constraint, ...
    'conRange', con_range, 'warpPara', warp_para));
u_opt.algo.maxEval   = checkSetInput(u_opt.algo, 'maxEval', 'i,>0', 100);
max_eval_u_0          = u_opt.algo.maxEval;
max_eval_u_df         = u_opt.algo.maxEval;
u_opt.algo.dataCast  = checkSetInput(u_opt.algo, 'dataCast', ...
    {'single','double','gpuArray-single','gpuArray-double'}, 'double');
u_opt.algo.outputPreStr = [output_pre_str checkSetInput(u_opt.algo, 'outputPreStr', 'char', '   ')];
switch u_solver
    case 'PDHG'
        % intial values for the dual variables and projections
        u_opt.algo.yU1            = checkSetInput(u_opt.algo, 'yU1', 'numeric', zeros(sz_v));
        u_opt.algo.yU2            = checkSetInput(u_opt.algo, 'yU2', 'numeric', zeros(sz_u));
        u_opt.algo.theta          = checkSetInput(u_opt.algo, 'theta', 'numeric', 1);
        u_opt.algo.preconditionFL = checkSetInput(u_opt.algo, 'preconditionFL', 'logical', true);
        u_opt.algo.tauSigmaFac    = checkSetInput(u_opt.algo, 'tauSigmaFac', 'numeric', 1);
    case 'ADMM'
        u_opt.ADMMVar            = checkSetInput(u_opt,'ADMMVar','struct',emptyStruct);
        u_opt.algo.sqNormF       = sqnorm_f;
        u_opt.algo.computeEnergy = true;
        u_opt.algo.returnAlgoVar = warmstart_u;
        u_opt.algo.lsSolverPara  = checkSetInput(u_opt.algo, 'lsSolverPara', 'struct', emptyStruct);
        u_ls_para_reset          = u_opt.algo.lsSolverPara;
    otherwise
        notImpErr
end


%%% parameter settings for the for the u update algorithm
% copy settings for the energy function
v_opt = mergeStructs(v_opt, struct('TVtype', TV_type_v, ...
    'OFtype', OF_type, 'p', p, 'symmetricOF', symmetric_OF, 'warpPara', warp_para));
% set some defaults
v_solver    = checkSetInput(v_opt, 'solver', {'PDHG','ADMM'}, 'ADMM');
v_mode      = checkSetInput(v_opt, 'mode', {'block','FbF'}, 'FbF');
warmstart_v = checkSetInput(v_opt, 'warmstart', 'logical', true);
v_opt.output        = checkSetInput(v_opt, 'output', 'logical', false);
switch OF_type
    case 'linear'
        % nothing to be done here
    case 'nonLinear'
        % parameters of the motion estimation pyramid
        v_opt.minSz            = checkSetInput(v_opt, 'minSz', 'i,>0', 4);
        v_opt.downSamplinggFac = checkSetInput(v_opt, 'downSamplinggFac', '>0', 0.5);
        v_opt.maxWarp          = checkSetInput(v_opt, 'maxWarp', 'i,>0', 3);
        v_opt.warpTol          = checkSetInput(v_opt, 'warpTol' , '>0', 10^-2);
        v_opt.computeEnergy    = true;
        v_opt.returnBest       = true;
        v_opt.levelIni         = 'best';
        v_opt.lineSearch       = true;
        v_opt.lineSearchMode   = 'coarse';
end
% overwrite some defaults for the solver
v_opt.algo         = checkSetInput(v_opt, 'algo', 'struct', emptyStruct, [], true);
% copy settings for the energy function into solver para
v_opt.algo = mergeStructs(v_opt.algo, struct('TVtype', TV_type_v, ...
    'OFtype', OF_type, 'p', p, 'symmetricOF', symmetric_OF, 'warpPara', warp_para));
v_opt.algo.maxEval = checkSetInput(v_opt.algo, 'maxEval', 'i,>0', 100);
max_eval_v_0       = v_opt.algo.maxEval;
max_eval_v_df      = v_opt.algo.maxEval;

v_opt.algo.dataCast = checkSetInput(v_opt.algo, 'dataCast', ...
    {'single','double','gpuArray-single','gpuArray-double'}, 'double');
v_opt.algo.outputPreStr  = [output_pre_str checkSetInput(v_opt.algo, 'outputPreStr', 'char', '   ')];
v_opt.algo.output        = checkSetInput(v_opt.algo, 'output', 'logical', false);
v_opt.algo.returnAlgoVar = checkSetInput(v_opt.algo, 'returnAlgoVar', 'logical', warmstart_v);
switch v_mode
    case 'block'
        oldImpErr
        switch v_solver
            case 'PDHG'
                % intial values for the dual variables and projections
                v_opt.yV = checkSetInput(v_opt, 'yV', 'numeric', zeros([n_xyz,dim_space,dim_space,n_t-1]));
                % initial values for PDHG step sizes
                v_opt.tau   = checkSetInput(v_opt, 'tau', 'numeric', 1/(dim_space*2));
                v_opt.sigma = checkSetInput(v_opt, 'sigma', 'numeric', 0.5);
                v_opt.theta = checkSetInput(v_opt, 'theta', 'numeric', 1);
                if(v_opt.tau * v_opt.sigma > 1/(4*dim_space))
                    warning('chosen step sizes might be too large')
                end
            case 'ADMM'
                v_opt.ADMMVar      = checkSetInput(v_opt, 'ADMMVar', 'struct', emptyStruct);
                v_opt.computeEnergy = true;
                v_opt.lsSolverPara = checkSetInput(v_opt, 'lsSolverPara', 'struct', emptyStruct);
                v_opt.rhoAdaptation = checkSetInput(v_opt, 'rhoAdaptation', 'logical', false);
                v_ls_para_reset  = v_opt.lsSolverPara;
                v_rho_adaptation = v_opt.rhoAdaptation;
            otherwise
                notImpErr
        end
        if(warmstart_v)
            v_opt.v = v;
        end
    case 'FbF'
        v_opt.maxEvalFactorFbF = checkSetInput(v_opt, 'maxEvalFactorFbF', 'i,>0', 10);
        
        u1 = cell(n_t-1,1);
        u2 = u1;
        v_opt_t         = cell(n_t-1,1);
        
        switch v_solver
            case 'ADMM'
                v_opt.algo.lsSolverPara  = checkSetInput(v_opt.algo, 'lsSolverPara', 'struct', []);
                v_opt.algo.rho           = checkSetInput(v_opt.algo, 'rho', '>0', 1);
                v_opt.algo.rhoAdaptation = checkSetInput(v_opt.algo, 'rhoAdaptation', 'logical', false);
                v_ls_para_reset   = v_opt.algo.lsSolverPara;
                v_rho_adaptation = v_opt.algo.rhoAdaptation;
                [z_fbF,~,v_opt] = checkSetInput(v_opt, 'zFbF', 'cell', cell(n_t-1,1), true);
                [w_fbF,~,v_opt] = checkSetInput(v_opt, 'wFbF', 'cell', cell(n_t-1,1), true);
                if(isscalar(v_opt.algo.rho))
                    v_opt.algo.rho = v_opt.algo.rho * ones(n_t-1,1);
                end
            case 'PDHG'
                [y_v_fbF,~,v_opt] = checkSetInput(v_opt, 'yVFbF', 'cell', cell(n_t-1,1), true);
        end
        warmstart_v = true;
        parallel_v  = checkSetInput(v_opt, 'parallel', 'logical', false);
        if(parallel_v)
            %%% open a parallel pool of workers
            warning off all
            df_pool_size     = maxNumCompThreads;
            warning on all
            n_worker_pool      = min(n_t-1,checkSetInput(v_opt, 'nWorkerPool', 'i,>0', df_pool_size));
            openParPool(n_worker_pool, false, false);
            v_opt.nWorker = Inf;
        else
            v_opt.nWorker = 0;
        end
        if(monotone_energy)
            v_opt.stopCriterion  = 'energyDec';
            v_opt.energyOverShot = energy_overshot;
        end
end


%%% define operators to convert precision of certain results
castFun = castFunction(u_opt.algo.dataCast, true);
% cast data here, otherwise it will be copied in sub-functions
f_or_ATf  = castFun(f_or_ATf);
switch u_opt.algo.dataCast
    case {'double', 'gpuArray-double'}
        switch v_opt.algo.dataCast
            case {'double', 'gpuArray-double'}
                upCast = @(x) x;
            case {'single', 'gpuArray-single'}
                upCast = @(x) double(x);
        end
    case {'single', 'gpuArray-single'}
        switch v_opt.algo.dataCast
            case {'double', 'gpuArray-double'}
                upCast = @(x) double(x);
            case {'single', 'gpuArray-single'}
                upCast = @(x) x;
        end
end


%%% line search parameter
if(line_search)
    max_eval_linesearch = checkSetInput(para, 'maxEvalLineSearch', 'i,>0', ...
        min(max_eval_u_df,max_eval_v_df));
end


%%% initialize energy terms and certain counters
energy_uv      = energyFun(u, v);
energy         = energy_uv;
min_energy     = energy_uv;
rel_energy_dec = Inf;
eval_u_total   = 0;
eval_v_total   = 0;
n_updates_u    = 0;
n_updates_v    = 0;
return_0F      = false;

% we need to store the best results achieved so far
if(~monotone_energy)
    u_return = u;
    v_return = v;
    %dataFitReturn = dataFit;
    %TVuReturn = TVu;
    %TVvReturn = TVv ;
    %optFlwDiscReturn = optFlwDisc;
    iter_return = 0;
end



% =========================================================================
% MAIN ITERATION
% =========================================================================


t_cmp_energy     = toc(clock_cmp_total);
t_cmp_u          = 0;
t_cmp_v          = 0;
t_cmp_linesearch = 0;

% initial output
if(output)
    last_output_energy = min_energy;
    disp([output_pre_str 'TVTVOpticalFlow Deblurring, start; minEnergy: ' num2str(last_output_energy,'%.6e')]);
    next_output_time = output_freq;
end

% main loop, will break if budget of iterations is used up or other
% condition fulfilled.
stop_main_loop = false;
while(~stop_main_loop)
    
    
    % =========================================================================
    % OPTIMIZE FOR U WHILE KEEPING V FIX
    % =========================================================================
    
    
    output_u = '';
    t_cmp_total = toc(clock_cmp_total);
    if(t_cmp_total > max_cmp_time)
        outputFun();
        break
    end
    eval_u_update = 0;
    eval_v_update = 0;
    t_cmp_u_iter  = 0;
    t_cmp_v_iter  = 0;
    
    % we need to store the iterates from the second but last iterations
    if(inertia_para > 0 && n_updates_u > 0)
        u_old_old = u_old;
        v_old_old = v_old;
    end
    % we need to store the iterates from the last iteration
    u_old = u;
    v_old = v;
    
    min_energy_old  = min_energy;
    energy_dec_flag = false;
    
    clock_cmp = tic;
    u_opt.algo.maxEval = max_eval_u_df;
    energy_this_update = [];
    
   
    gamma_here = gamma;
    update_u   = true;
    % modify the first u-Update
    if(n_updates_u == 0)
        switch u_ini_mode
            case 'skip'
                update_u    = false;
                energy_here = energy;
                t_cmp_total = toc(clock_cmp_total);
            case 'TV-FbF'
                gamma_here         = gamma * 10^-1;
                u_opt.algo.maxEval = 10 * max_eval_u_df;
            case 'TVOF'
                u_opt.algo.maxEval = 10 * max_eval_u_df;
            case 'normal'
                % nothing to be done
        end
    end
    
    % the while loop ensures a sufficient energy decay
    while(update_u)
        
        % call sub-algorithms
        switch u_solver
            case 'PDHG'
                oldImpErr
                %[u,uOptPara.u,uOptPara.yU1,uOptPara.yU2,infoU] = TVplusOpticalFlowDenosing_PDHG(fOrATf,v,alpha,gammaHere,uOptPara);
                %evalUUpdateHere = uOptPara.maxEval;
            case 'ADMM'
                [u, infoU] = TVOF_Deblurring_ADMM(A, AT, f_or_ATf, v, alpha, gamma_here, u_opt.algo);
                u_opt.algo.u            = u;
                u_opt.algo.ADMMVar      = infoU.ADMMVar;
                u_opt.algo.rho          = infoU.rho;
                u_opt.algo.lsSolverPara = overwriteFields(u_opt.algo.lsSolverPara, infoU.lsSolverPara, true);
                u                      = projectionU(u);
                eval_u_update_here        = infoU.evalVsIter(end);
            otherwise
                notImpErr
        end
        u                = upCast(u);
        energy_this_update = [energy_this_update(1:end-1); infoU.energy(:)];
        clear infoU
        
        %loglog(energyThisUpdate);title('u update');drawnow();
        
        % evaluate the proposed u update
        energy_here   = energyFun(u, v);
        eval_u_update  = eval_u_update + eval_u_update_here;
        t_cmp_total   = toc(clock_cmp_total);
        % check whether we need to perform another round s
        if(monotone_energy && energy_here >= energy_uv && (~hard_cmp_restr || t_cmp_total <= max_cmp_time))
            % go for another round
        elseif(~monotone_energy)
            u = u_opt.algo.u;
            break
        else % energy <=  minEnergy || tCompTotal > maxCompTime
            % ensure a sufficient decrease of energy
            if(hard_cmp_restr && t_cmp_total > max_cmp_time)
                break
            end
            first_energy_dec = find(energy_this_update(2:end) <= energy_this_update(1), 1, 'first') + 1;
            if(~isempty(first_energy_dec)  && first_energy_dec <= ceil(length(energy_this_update)/energy_overshot))
                break
            end
        end
    end
    if(warmstart_u)
        switch u_solver
            case 'PDHG'
                % nothing to be done
            case 'ADMM'
                u_opt.algo.u = u;
                u_opt.algo.lsSolverPara = u_ls_para_reset;
            otherwise
                notImpErr
        end
    else
        if(~any(u(:)))
            return_0F = true;
            break
        end
    end
    % update energy and evaluations
    energy      = [energy, energy_here];
    energy_uv    = energy_here;
    t_cmp_energy = [t_cmp_energy, t_cmp_total];
    eval_u_total  = eval_u_total + eval_u_update;
    n_updates_u   = n_updates_u + 1;
    checkEnergyAndReturnFun()
    
    % over-relaxation in u
    if(overrelax_para ~= 1)
        u = (1-overrelax_para) * u_old + overrelax_para * u;
        energy_uv = energyFunWithoutCon(u,v);
    end
    t_cmp_u_iter = toc(clock_cmp);
    t_cmp_u     = t_cmp_u + t_cmp_u_iter;
    
    
    % =========================================================================
    % OPTIMIZE FOR V WHILE KEEPING U FIXED
    % =========================================================================
    
    
    output_v = '';
    % check if u is non-zero
    if(~any(u(:)))
        notImpErr
    end
    
    t_cmp_total = toc(clock_cmp_total);
    if(t_cmp_total > max_cmp_time)
        outputFun();
        break
    end
    clock_cmp        = tic;
    v_opt.algo.maxEval = max_eval_v_df;
    energy_this_update = [];
    
    switch v_mode
        case 'block'
            outdatedCodeSection
%             % the while loop ensures a sufficient energy decay
%             while(1)
%                 % call sub-algorithms to compute v-Update
%                 switch vSolver
%                     case 'PDHG'
%                         [v,vOpt.v,vOpt.yV,infoV] = OpticalFlowTVMotionEstimation_PDHG(u,lambda,vOpt);
%                     case 'ADMM'
%                         [v,vOpt.v,infoV] = OpticalFlowTVMotionEstimation_ADMM(u,lambda,vOpt);
%                         vOpt.ADMMVar = infoV.ADMMVar;
%                         vOpt.rho     = infoV.rho;
%                         vOpt.lsSolverPara = overwriteFields(vOpt.lsSolverPara,infoV.lsSolverPara,true);
%                     otherwise
%                         notImpErr
%                 end
%                 vOpt.vBest = v;
%                 vOpt.minEnergy = infoV.minEnergy;
%                 vOpt.TVv = infoV.TVv;
%                 vOpt.optFlwDisc = infoV.optFlwDisc;
%                 infoV.energy = sum(infoV.energy,1);
%                 v = upCast(v);
%                 
%                 energyThisUpdate = [energyThisUpdate(1:end-1);infoV.energy(:)];
%                 
%                 %plot(energyThisUpdate);title('v update');drawnow();
%                 
%                 % evaluate the proposed v update
%                 % compute energy (without constraints as u might be
%                 % over-relaxed)
%                 energyHere      = energyFunWithoutCon(u,v);
%                 evalVUpdate     = evalVUpdate + vOpt.maxEval;
%                 tCompTotal      = toc(clockCompTotal);
%                 if(monotoneEnergy && energyHere >= energyUV && (~hardCompRestr || tCompTotal <= maxCompTime))
%                     % go for another round
%                     %elseif(updateAllFramesFL && any(infoV.iterBest(1:end-1) == 0))
%                     % go for another round
%                     %    notImpErr
%                     % not implemented properly, needs to add up iterBest
%                 elseif(~monotoneEnergy)
%                     v = vOpt.v;
%                     break
%                 else % energy < minEnergy
%                     % ensure a sufficient decrease of energy
%                     if(hardCompRestr && tCompTotal > maxCompTime)
%                         break
%                     end
%                     firstEnergyDec = find(energyThisUpdate(2:end) <= energyThisUpdate(1),1,'first') + 1;
%                     if(~isempty(firstEnergyDec)  && firstEnergyDec <= ceil(length(energyThisUpdate)/energyOverShot))
%                         break
%                     end
%                 end
%             end
%             if(warmstartV)
%                 vOpt.v = v;
%                 switch vSolver
%                     case 'PDHG'
%                         vOpt = myRmfield(vOpt,{'minEnergy','vBest','TVv','optFlwDisc','energy'});
%                     case 'ADMM'
%                         vOpt = myRmfield(vOpt,{'minEnergy','vBest','TVv','optFlwDisc','energy','iter'});
%                         vOpt.lsSolverPara  = vLSParaReset;
%                         vOpt.rhoAdaptation = vRhoAdaptation;
%                     otherwise
%                         notImpErr
%                 end
%             end

        case 'FbF'
            
            %%% frame by frame updates
            
            if(parallel_v)
                % open a pool if the existing timed out
                openParPool(n_worker_pool, false, false);
            end
            
            % prepare parfor loop over t
            for t = 1:n_t-1
                v_opt_t{t}       = v_opt;
                u1{t}            = sliceArray(u, dim_space+1, t,   1);
                u2{t}            = sliceArray(u, dim_space+1, t+1, 1);
                switch v_solver
                    case 'PDHG'
                        notImpErr
                    case 'ADMM'
                        v_opt_t{t}.algo.rho = v_opt.algo.rho(t);
                end
            end
            v = spaceTime2DynamicData(v, sz_v);
            info_v_t = cell(n_t-1,1);
                    
            switch OF_type
                
                %%% linearlized optical flow equation
                case 'linear'
                    
                    if(symmetric_OF)
                        notImpErr
                    end
                    
                    % each subroutine handles the management of minimal and
                    % maximal evaluations
                    v_opt.algo.minEval = v_opt.algo.maxEval;
                    v_opt.algo.maxEval = v_opt.maxEvalFactorFbF * v_opt.algo.maxEval;
                    
                    switch v_solver
                        case 'PDHG'
                            oldImpErr
%                             switch dimSpace
%                                 case 2
%                                     parfor(t = 1:nTime-1, vOpt.nWorker)
%                                         [v(:,:,t,:), yVFbF{t}, infoVT{t}] = ...
%                                             OFTV_MotionEstimation_PDHG2Frames(u1{t}, u2{t}, squeeze(v(:,:,t,:)),...
%                                             yVFbF{t}, lambda, vOptT{t});
%                                         u1{t} = [];
%                                         u2{t} = [];
%                                     end
%                                 case 3
%                                     parfor(t = 1:nTime-1, vOpt.nWorker)
%                                         [v(:,:,:,t,:), yVFbF{t}, infoVT{t}] = ...
%                                             OFTV_MotionEstimation_PDHG2Frames(u1{t}, u2{t}, squeeze(v(:,:,:,t,:)),...
%                                             yVFbF{t}, lambda, vOptT{t});
%                                         u1{t} = [];
%                                         u2{t} = [];
%                                     end
%                                 otherwise
%                                     notImplErr
%                             end
                        case 'ADMM'
                            
                            parfor(t = 1:n_t-1,v_opt.nWorker)
                                a = [];
                                switch dim_space
                                    case 2
                                        [gradU2y, gradU2x] = gradient(u2{t});
                                        a  = cat(3, gradU2x, gradU2y);
                                    case 3
                                        [gradU2y, gradU2x, gradU2z] = gradient(u2{t});
                                        a  = cat(4, gradU2x, gradU2y, gradU2z);
                                end
                                
                                ut = - (u2{t} - u1{t});
                                u1{t} = [];
                                u2{t} = [];
                                
                                [v{t}, z_fbF{t}, w_fbF{t}, info_v_t{t}] = ...
                                    TV_FlowEstimation_ADMM(a, ut,...
                                    v{t}, z_fbF{t}, w_fbF{t}, lambda(t), v_opt_t{t}.algo);
                            end
                            
                    end
                    
                    % merge results back
                    info_v      = [];
                    energy_cell = cell(n_t-1, 1);
                    max_length  = 0;
                    
                    for t = 1:n_t-1
                        energy_cell{t}   = info_v_t{t}.energy;
                        max_length       = max(max_length,length(energy_cell{t}));
                        info_v.eval(t)   = info_v_t{t}.nEval;
                    end
                    
                    % adjust energy to same length
                    for t = 1:n_t-1
                        if(length(energy_cell{t}) < max_length)
                            energy_cell{t} = [energy_cell{t},energy_cell{t}(end) * ones(1,max_length-length(energy_cell{t}))];
                        end
                    end
                    info_v.energy = cell2mat(energy_cell);
                    
                    %plot(bsxfun(@minus, infoV.energy, infoV.energy(:,1))'); title('v update'); drawnow();
                    
                    switch v_solver
                        case 'PDHG'
                            notImpErr
                        case 'ADMM'
                            for t = 1:n_t-1
                                v_opt.rho(t) = info_v_t{t}.rho;
                            end
                    end
                    
                case 'nonLinear'
                    
                    parfor(t = 1:n_t-1, v_opt.nWorker)
                    %for t=1:nT-1   
                        %vOptT{t}.output = true;
                        %vOptT{t}.algo.output = true;
                        if(warmstart_v || monotone_energy)
                            v_opt_t{t}.v = v{t};
                        end
                        v{t} = opticalFlowEstimation(u1{t}, u2{t}, lambda(t), v_opt_t{t});
                    end
                    
                    info_v      = [];
                    info_v.eval = v_opt.algo.maxEval;
                    
            end
            v = dynamicData2SpaceTime(v);
            v = upCast(v);
            
            
            energy_here   = energyFunWithoutCon(u, v);
            eval_v_update  = eval_v_update + max(info_v.eval);
            t_cmp_total   = toc(clock_cmp_total);
    end
    % update energy and evaluations
    energy      = [energy, energy_here];
    energy_uv    = energy_here;
    t_cmp_energy = [t_cmp_energy, t_cmp_total];
    eval_v_total  = eval_v_total + eval_v_update;
    n_updates_v   = n_updates_v + 1;
    checkEnergyAndReturnFun()
    
    
    % =========================================================================
    % REMAINING OPERATIONS AND OUTPUT
    % =========================================================================
    
    
    %%% over-relaxation
    if(overrelax_para ~= 1)
        oldImpErr % warmstart of v needs to be adapted
        u        = projectionU(u);
        v        = (1-overrelax_para) * v_old + overrelax_para * v;
        energy_uv = energyFun(u,v);
        
        if(warmstart_u)
            u_opt.algo.u = u;
        end
        if(warmstart_v)
            v_opt.algo.v = v;
        end
    end
    
    t_cmp_v_iter = toc(clock_cmp);
    t_cmp_v = t_cmp_v + t_cmp_v_iter;
    
    
    %%% inertia
    t_cmp_total = toc(clock_cmp_total);
    if((t_cmp_total > max_cmp_time) && hard_cmp_restr)
        outputFun();
        break
    end
    if(inertia_para > 0 && n_updates_v > 1)
        oldImpErr % warmstart of v needs to be adapted
        u          = projectionU(u + inertia_para * (u_old - u_old_old));
        v          = v + inertia_para * (v_old - v_old_old);
        t_cmp_total = toc(clock_cmp_total);
        
        t_cmp_energy = [t_cmp_energy, t_cmp_total];
        energy_uv    = energyFun(u, v);
        energy      = [energy, energy_uv];
        checkEnergyAndReturnFun()
        
        if(warmstart_u)
            u_opt.algo.u = u;
        end
        if(warmstart_v)
            v_opt.algo.v = v;
        end
    end
    
    
    %%% line search
    t_cmp_total = toc(clock_cmp_total);
    if((t_cmp_total > max_cmp_time) && hard_cmp_restr)
        outputFun();
        break
    end
    if(line_search)
        oldImpErr % warmstart of v needs to be adapted
        lineSearchFun = @(s) energyFun(projectionU(u_old + s * (u-u_old)),v_old + s * (v-v_old));
        
        ub = 2;
        lb = -0.1;
        linesearch_opt = optimset('Display','none', 'MaxFunEvals', max_eval_linesearch, 'TolX', 1e-9);
        
        clock_cmp = tic;
        [s_opt,energy_uv] = fminbnd(lineSearchFun, lb, ub, linesearch_opt);
        u               = projectionU(u_old + s_opt * (u - u_old));
        v               = v_old + s_opt * (v-v_old);
        % recompute the energy such that it is consistent with other
        % routines
        energy      = [energy, energy_uv];
        t_cmp_total  = toc(clock_cmp_total);
        t_cmp_energy = [t_cmp_energy, t_cmp_total];
        checkEnergyAndReturnFun()
        t_cmp_linesearch = t_cmp_linesearch + toc(clock_cmp);
        
        if(warmstart_u)
            u_opt.algo.u = u;
        end
        if(warmstart_v)
            u_opt.algo.v = v;
        end
    end
    
    
    %%% adjust computational efford in sub-optimizations
    %     switch adjustMaxEval
    %         case 'compTime'
    %             if(tCompV > tCompU * 1.5)
    %                 maxEvalV_df = maxEvalV_0;
    %                 maxEvalU_df = ceil(1.5 * maxEvalU_df);
    %             elseif (tCompU > tCompV * 1.5)
    %                 maxEvalU_df = maxEvalU_0;
    %                 maxEvalV_df = ceil(1.5 * maxEvalV_df);
    %             end
    %         case 'energyDec'
    %             energyDec = abs(diff(energy));
    %             energyDecU = energyDec(1:2:end);
    %             energyDecV = energyDec(2:2:end);
    %             % weight such that newer iterates are more important, sum up and
    %             % divide by computation time spend
    %             energyDecU = sum(energyDecU./(2.^((nUpdatesU-1):-1:0)))/tCompU;
    %             energyDecV = sum(energyDecV./(2.^((nUpdatesV-1):-1:0)))/tCompV;
    %             if(energyDecU > energyDecV * 10)
    %                 maxEvalV_df = maxEvalV_0;
    %                 maxEvalU_df = min(10*maxEvalU_0,ceil(2 * maxEvalU_df));
    %             elseif (energyDecV > energyDecU * 10)
    %                 maxEvalU_df = maxEvalU_0;
    %                 maxEvalV_df = min(10*maxEvalV_0,ceil(2 * maxEvalV_df));
    %             else
    %                 maxEvalU_df = max(ceil(maxEvalU_df/2),maxEvalU_0);
    %                 maxEvalV_df = max(ceil(maxEvalV_df/2),maxEvalV_0);
    %             end
    %         case 'half'
    %             maxEvalU_df = ceil(maxEvalU_df/2);
    %             maxEvalV_df = ceil(maxEvalV_df/2);
    %     end
    
    %%% check for convergence criteria
    switch stop_criterion
        case 'maxIter'
            stop_main_loop = (n_updates_u + n_updates_v)/2 == max_iter;
        case 'relEnergy'
            rel_energy_dec = (min_energy_old - min_energy)/(min_energy_old + min_energy);
            stop_main_loop = (n_updates_u + n_updates_v)/2 >= min_iter  && energy_dec_flag && rel_energy_dec < stop_tol;
        case 'compTime'
            t_cmp_total = toc(clock_cmp_total);
            stop_main_loop = (n_updates_u + n_updates_v)/2 >= min_iter  && t_cmp_total > max_cmp_time;
    end
    stop_main_loop = stop_main_loop | (n_updates_u + n_updates_v)/2 == max_iter;
    
    %%% print output (via nested function)
    outputFun();
end

t_cmp_total = toc(clock_cmp_total);


% =========================================================================
% GATHER RESULTS AND CLEAN UP
% =========================================================================


%%% print final output
if(output)
    output_str = [output_pre_str 'TVTVOpticalFlowDenoising, finished. total comp time U/V: ' ...
        convertSec(t_cmp_u) '/' convertSec(t_cmp_v)];
    disp(output_str)
end


%%% prepare all variables that need to be returned
if(monotone_energy || return_0F)
    u_return    = u;
    v_return    = v;
    iter_return = [n_updates_u, n_updates_v];
end

u_return = projectionU(u_return);

info.tCompV      = t_cmp_v;
info.tCompU      = t_cmp_u;
info.nUpdatesU   = n_updates_u;
info.nUpdatesV   = n_updates_v;
info.iterUTotal  = eval_u_total;
info.iterVTotal  = eval_v_total;
info.iterReturn  = iter_return;
info.energy      = double(gather(energy));
info.tCompEnergy = t_cmp_energy;
info.dataFit     = dataFitFun(u_return);
info.TVu         = TVuFun(u_return);
info.TVv         = TVvFun(v_return);
info.optFlwDisc  = OFfun(u_return,v_return);

if(return_opt_var)
    info.uOpt         = u_opt;
    info.vOpt         = v_opt;
    % reset maxEvals
    info.uOpt.maxEval = max_eval_u_0;
    info.vOpt.maxEval = max_eval_v_0;
    switch v_mode
        case 'FbF'
            switch v_solver
                case 'PDHG'
                    info.vOpt.yVFbF = y_v_fbF;
                case 'ADMM'
                    info.vOpt.zFbF  = z_fbF;
                    info.vOpt.wFbF  = w_fbF;
            end
    end
end

info  = orderfields(info);

switch v_mode
    case 'FbF'
        if(parallel_v)
            closeParPool(gcp,false);
        end
end



% =========================================================================
% NESTED FUNCTIONS
% =========================================================================


%%% function to print output
    function outputFun()
        rel_change_u = norm(u(:) - u_old(:)) / norm(u(:));
        rel_change_v = norm(v(:) - v_old(:)) / norm(v(:));
        
        if(output && t_cmp_total >= next_output_time)
            output_str = [output_pre_str 'TVTVOF, it ' num2str((n_updates_u+n_updates_v)/2)];
            switch stop_criterion
                case 'maxIter'
                    output_str = [output_str '/' int2str(max_iter)];
            end
            output_str = [output_str ':'];
            switch stop_criterion
                case 'relEnergy'
                    output_str = [output_str ' minE: ' num2str(min_energy, '%.4e') ...
                        ' (dec: ' num2str(rel_energy_dec,'%.4e') ')'];
                otherwise
                    output_str = [output_str ' minE: ' num2str(min_energy, '%.4e') ...
                        ' (dec: ' num2str((last_output_energy-min_energy)/min_energy, '%.4e') ')'];
            end
            output_str = [output_str '; rCh U/V: ' num2str(rel_change_u, '%.3e') '/' num2str(rel_change_v, '%.3e')];
            output_str = [output_str '; eval U/V: ' int2str(eval_u_update) '/' int2str(eval_v_update)];
            output_str = [output_str '; cmpT U/V: ' convertSec(t_cmp_u_iter) '/' convertSec(t_cmp_v_iter)];
            if(~isinf(max_cmp_time))
                output_str = [output_str ' (tot:' convertSec(t_cmp_total) '/' convertSec(max_cmp_time) ')'];
            end
            if(~isempty(output_u))
                output_str = [output_str '; ' output_u];
            end
            if(~isempty(output_v))
                output_str = [output_str '; ' output_v];
            end
            
            disp(output_str)
            last_output_energy = min_energy;
            next_output_time   = next_output_time + output_freq;
        end
    end

%%% function for checking energy of new (u,v) pair and setting return
%%% variables
    function checkEnergyAndReturnFun()
        if(energy_here < min_energy)
            energy_dec_flag = true;
            min_energy =  energy_here;
            if(~monotone_energy)
                u_return    = u;
                v_return    = v;
                iter_return = [n_updates_u, n_updates_v];
            end
        end
    end
end