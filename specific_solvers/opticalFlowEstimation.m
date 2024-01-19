function [v_return, info] = opticalFlowEstimation(u1, u2, beta, para)
%OPTICALFLOWESTIMATION estimates the optical flow between two images
%
% DETAILS:
%   tries to solve the underdetermined non-linear optical flow equation
%   u2(x + v) = u1(x) or more precisely warpImage(u2, v) = u1
%   for v in a regularized way by solving the non-convex variational problem
%   v = argmin_v  1/p || warpImage(u2, v) - u1 ||_p^p + beta J(v),
%   where p is in [1,2], beta > 0 and J is a regularization functional such
%   as total variation. For the solution, a coarse-to-fine pyramid with
%   multiple linearizations of warpImage(u2, v) and line-search is used.
%
% USAGE:
%   [v, info] = opticalFlowEstimation(u1, u2, beta, para)
%
% INPUTS:
%   u1   - 2D or 3D numerical array
%   u2   - 2D or 3D numerical array
%   beta - non-negative regularization parameter
%
% OPTIONAL INPUTS:
%   para - a struct containing further optional parameters:
%       TODO
%       'interpolationMethod' - see interpn.m, default: linear
%
% OUTPUTS:
%   v_return - estimated motion field statisfying the optical flow equation
%   info    - struct containing additional parameters
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 17.12.2018
%       last update     - 27.10.2023
%
% See also

% check user defined value for para, otherwise assign default value
if(nargin < 4)
    para = [];
end


dim_u = nDims(u1);
sz_u  = size(u1);

% output parameter
output           = checkSetInput(para, 'output', 'logical', false);
output_pre_str   = checkSetInput(para, 'outputPreStr', 'char', '');
compute_energy   = checkSetInput(para, 'computeEnergy', 'logical', true);
return_best      = checkSetInput(para, 'returnBest', 'logical', true);

% paramters of the variational energy 
symmetric_OF      = checkSetInput(para, 'symmetricOF', 'logical', false);
spatial_grad_type = checkSetInput(para, 'spatialGrad', {'central'}, 'central');
p                 = checkSetInput(para, 'p', '>0', 2);
TV_type           = checkSetInput(para, 'TVtype',...
                    {'anisotropic','mixedIsotropic','fullyIsotropic'}, 'mixedIsotropic');

% parameters of the coarse-to-fine warping pyramid
level_ini        = checkSetInput(para, 'levelIni', {'best', 'last'}, 'best');
ds_fac           = checkSetInput(para, 'downSamplinggFac' , '>0', 0.5);
ds_iterative     = checkSetInput(para, 'downSampleIterative' , 'logical', true);
[n_level, df]    = checkSetInput(para, 'nLevel', 'i,>0', 1);
if(df)
    min_sz       = checkSetInput(para, 'minSz', 'i,>0', 6);
    n_level      = ceil(log(min_sz/max(sz_u))/log(ds_fac)) + 1;
end
max_warp          = checkSetInput(para, 'maxWarp' , 'i,>0', 1);
warp_tol          = checkSetInput(para, 'warpTol' , '>0', 10^-3);
interp_method     = checkSetInput(para, 'interpolationMethod', ...
                    {'linear', 'nearest', 'pchip', 'cubic', 'spline', 'makima'}, 'linear');
smoothing_kernel  = checkSetInput(para, 'smoothingKernel' , ...
                    {'Gaussian', 'none'}, 'Gaussian');

% parameters of the line-search algorithm
lns               = checkSetInput(para, 'lineSearch', 'logical', true);
lns_mode          = checkSetInput(para, 'lineSearchMode', {'coarse', 'original'}, 'coarse');
lns_iter          = checkSetInput(para, 'lineSearchIter', 'i,>0', 10);
add_neg_dir       = checkSetInput(para, 'lineSearchAddNegDir', 'logical', true);
nu_0              = checkSetInput(para, 'nu', '>0', 1);
lns_para          = [];
lns_para.maxEval  = lns_iter;
lns_para.type     = 'fminbnd';
lns_para.storeBest= true;

% general parameters
data_cast         = checkSetInput(para, 'dataCast',...
                    {'single','double','gpuArray-single','gpuArray-double'}, 'double');

% parameter of the solverI = imfilter(I_ori, mask, 'replicate');
solver            = checkSetInput(para, 'solver', {'ADMM'}, 'ADMM');
algo              = checkSetInput(para, 'algo', 'struct', []);
algo.outputPreStr = checkSetInput(algo, 'outputPreStr', 'char', '   ');
algo.outputPreStr = [output_pre_str algo.outputPreStr];


% prepare warping and scaling parameter
wrp_sc_para = [];
wrp_sc_para.interpolationMethod = interp_method;

% prepare smoothing
switch smoothing_kernel
    case 'Gaussian'
        smooth = @(u, dsFac) imfilter(u, gaussianKernel(dim_u, 1./sqrt(2 * dsFac), 3), 'replicate');
    case 'none'
        smooth = @(u) u;
    otherwise
        notImpErr
end


% prepare parameters of the solver
switch solver
    case 'ADMM'
        % copy certain parameters
        algo.p              = p;
        algo.TVtype         = TV_type;
        algo.symmetricOF    = symmetric_OF;
        rho_ini             = checkSetInput(algo, 'rho', '>0', 1);
    otherwise
        notImpErr
end


% get a type casting function
[castFun, castZeros] = castFunction(data_cast);

% initial guess for v
[v, df_v]  = checkSetInput(para, 'v' , 'numeric', castZeros([sz_u dim_u]));
if(compute_energy)
    energy              = zeros(n_level, max_warp + 1);
    energy_aux_problems = zeros(n_level, max_warp + 1);
    if(df_v)
        min_energy = 1/p *  sumAll(abs(u1 - u2).^p);
    else
        min_energy = cmpEnergy(u1, u2, v);
    end
    if(output)
        disp([output_pre_str 'Initial energy: ' num2str(min_energy, '%.2e') '.'])
    end
end
v_return = v;

% cast u1 and u2 and prepare pyramids
u1_pyramid   = {castFun(u1)};
u2_pyramid   = {castFun(u2)};
sz_u_pyramid = {sz_u};
t_cmp        = zeros(n_level, max_warp);
v_max_change = zeros(n_level, max_warp);

%%% prepare the image pyramid by smoothing and interpolation
if(output && n_level > 1)
    fprintf([output_pre_str 'Prepare coarse-to-fine pyramid...'])
end

clock_prepare = tic;
for i_level = 2:n_level
    
    % compute image size
    sz_u_coarse           = ceil(sz_u * ds_fac^(i_level-1)); % size of coarse image
    sz_u_pyramid{i_level} = sz_u_coarse;
    
    % smoothing
    if(ds_iterative)
        u1_pyramid{i_level} = smooth(u1_pyramid{i_level-1}, sz_u_coarse./sz_u_pyramid{i_level-1});
        u2_pyramid{i_level} = smooth(u2_pyramid{i_level-1}, sz_u_coarse./sz_u_pyramid{i_level-1});
    else
        u1_pyramid{i_level} = smooth(u1_pyramid{1}, sz_u_coarse./sz_u_pyramid{1});
        u2_pyramid{i_level} = smooth(u2_pyramid{1}, sz_u_coarse./sz_u_pyramid{1});
    end
    
    % down sampling by interpolation
    u1_pyramid{i_level} = scaleImage(u1_pyramid{i_level}, sz_u_coarse, wrp_sc_para);
    u2_pyramid{i_level} = scaleImage(u2_pyramid{i_level}, sz_u_coarse, wrp_sc_para);
end
% down sample initial guess for v
v = scaleMotion(smooth(v, sz_u_pyramid{end}./sz_u_pyramid{1}), sz_u_pyramid{end});
t_prepare = toc(clock_prepare);


myDisp([int2str(n_level) ' level, max ' int2str(max_warp) ...
    ' warps' '; computation time: ' convertSec(t_prepare) '.'], output && n_level > 1)


%%% solve pyramid
for i_level = n_level:-1:1
    
    myDisp([output_pre_str 'Solve level ' int2str(i_level) '/' int2str(n_level) ...
            '; image size [' int2str(sz_u_pyramid{i_level}) ']; warps: 1'], ...
            output && n_level > 1, false)

    % upscale v and adjust vector length
    if(i_level < n_level)
        switch level_ini
            case 'best'
                v = scaleMotion(v_return, sz_u_pyramid{i_level});
            case 'last'
                v = scaleMotion(v, sz_u_pyramid{i_level});      
        end
    end
    
    % re-set step length 
    nu = nu_0;
    
    % compute energy on coarse and fine scale
    if(compute_energy)
        energy_aux_problems(i_level, 1) = cmpEnergy(u1_pyramid{i_level}, u2_pyramid{i_level}, v);
        energy(i_level, 1)              = cmpEnergy(u1, u2, v);
        if(return_best && energy(i_level, 1) < min_energy)
            v_return = v;
        end
        min_energy = min(min_energy, energy(i_level, 1));
    end
    
    i_warp = 0;
    continue_warping = true;
    
    while(i_warp < max_warp && continue_warping)
        
        clock_cmp = tic;
        i_warp = i_warp + 1;
           
        
        %%% compute terms appearing in the linearized problem
        % warp u2
        u2_tilde = warpImage(u2_pyramid{i_level}, v, [], wrp_sc_para);
        % compute its spatial gradients to compute a
        switch spatial_grad_type
            case 'central'
                switch dim_u
                    case 2
                        [grad{2}, grad{1}] = gradient(u2_tilde);
                    case 3
                        [grad{2}, grad{1}, grad{3}] = gradient(u2_tilde);
                end
                a = cat(dim_u+1, grad{:});
            otherwise
                notImpErr
        end
        % set up right hand side
        f = - (u2_tilde - u1_pyramid{i_level}) +  sum(a .* v, dim_u+1);
        % compute additional terms for symmetrical optical flow
        if(symmetric_OF)
            % warp u1
            u1_tilde = warpImage(u1_pyramid{i_level}, -v, [], wrp_sc_para);
            switch spatial_grad_type
                case 'central'
                    switch dim_u
                        case 2
                            [grad{2}, grad{1}] = gradient(u1_tilde);
                        case 3
                            [grad{2}, grad{1}, grad{3}] = gradient(u1_tilde);
                    end
                    a_sym = -cat(dim_u+1, grad{:});
                otherwise
                    notImpErr
            end
            f_sym = - (u1_tilde - u2_pyramid{i_level}) +  sum(a_sym .* v, dim_u+1);
        end
        
        
        %%% minimize variational problem with linearized optical flow term
        switch solver
            case 'ADMM'
                
                % call TV_FlowEstimation_ADMM
                if(i_warp == 1)
                    if(i_level == n_level)
                        z = []; w = [];
                    else
                        sc_vec_fac = sz_u_pyramid{i_level}./sz_u_pyramid{i_level+1};
                        if(iscell(z))
                            z{1} = scaleVectorField(scaleImage(z{1}, [sz_u_pyramid{i_level}, dim_u, dim_u], wrp_sc_para), sc_vec_fac, dim_u+1);
                            w{1} = scaleVectorField(scaleImage(w{1}, [sz_u_pyramid{i_level}, dim_u, dim_u], wrp_sc_para), sc_vec_fac, dim_u+1);
                            z{2} = scaleImage(z{2}, sz_u_pyramid{i_level}, wrp_sc_para);
                            w{2} = scaleImage(w{2}, sz_u_pyramid{i_level}, wrp_sc_para);
                            if(symmetric_OF)
                                z{3} = scaleImage(z{3}, sz_u_pyramid{i_level}, wrp_sc_para);
                                w{3} = scaleImage(w{3}, sz_u_pyramid{i_level}, wrp_sc_para);
                            end
                        else
                            z = scaleVectorField(scaleImage(z, [sz_u_pyramid{i_level}, dim_u, dim_u], wrp_sc_para), sc_vec_fac, dim_u+1);
                            w = scaleVectorField(scaleImage(w, [sz_u_pyramid{i_level}, dim_u, dim_u], wrp_sc_para), sc_vec_fac, dim_u+1);
                        end
                    end
                    algo.rho = rho_ini;
                end
                
                if(symmetric_OF)
                    [v_new, z, w, info.algoInfo{i_level,i_warp}] = ...
                        TV_SymmetricFlowEstimation_ADMM(a, a_sym, f, f_sym, v, z, w, beta, algo);
                else
                    [v_new, z, w, info.algoInfo{i_level,i_warp}] = ...
                        TV_FlowEstimation_ADMM(a, f, v, z, w, beta, algo);
                end
                
                algo.rho = info.algoInfo{i_level,i_warp}.rho;
            otherwise
                notImpErr
        end
        
        
        %%% perform line search in direction of the solution found
        search_direction = v_new - v; clear v_new;
        if(lns)
            switch lns_mode
                case 'coarse'
                    lnsFun    = @(v) cmpEnergy(u1_pyramid{i_level}, u2_pyramid{i_level}, v);
                    lns_ini_energy = energy_aux_problems(i_level, i_warp);
                case 'original'
                    lnsFun    = @(v) cmpEnergy(u1, u2, v);
                    lns_ini_energy  = energy(i_level, i_warp);
            end
            lns_para.Fx0 = lns_ini_energy;
            [nu, v] = linesearch(v, search_direction, lnsFun, nu, lns_para);
            
            if(nu == 0 && add_neg_dir)
               % see if a beter solution can be reached in opposite direction 
               for i_neg = 1:lns_iter
                   if(lnsFun(v - 10^(-i_neg) * search_direction) < lns_ini_energy)
                       nu = 10^(-i_neg);
                       v  = v - nu * search_direction;
                       break 
                   end
               end
            end
        else
            v = v + nu * search_direction;
        end
        
        
        %%% compute change in v and print output
        v_max_change(i_level, i_warp) = nu * maxAll(abs(search_direction));
        continue_warping           = v_max_change(i_level, i_warp) > warp_tol;
        
        t_cmp(i_level, i_warp) = toc(clock_cmp);
        
        if(compute_energy)
            energy_aux_problems(i_level, i_warp + 1) = cmpEnergy(u1_pyramid{i_level}, u2_pyramid{i_level}, v);
            energy(i_level, i_warp + 1) = cmpEnergy(u1, u2, v);
            if(return_best && energy(i_level, i_warp + 1) < min_energy)
                v_return = v;
            end
            min_energy = min(min_energy, energy(i_level, i_warp + 1));
        else
            v_return = v;
        end
        
        if(output && n_level > 1)
            if(i_warp < max_warp && continue_warping)
                fprintf([', ' int2str(i_warp+1)])
            else
                outstr = ['; computation time: ' convertSec(sumAll(t_cmp(i_level, :)))];
                if(compute_energy)
                    outstr = [outstr '; min energy: ' num2str(min(energy(i_level, 1:i_warp+1)), '%.2e') '.'];
                else
                    outstr = [outstr '.'];
                end
                disp(outstr)
            end
        end
        
        
    end
    
end
myDisp([output_pre_str 'Total computation time: ' convertSec(sum(t_cmp(:)) + t_prepare) '.'], output);


% scale if not in correct size already
v_return = scaleMotion(v_return, sz_u);

info.tPrepare   = t_prepare;
info.tComp      = t_cmp;
info.vMaxChange = v_max_change;
if(compute_energy)
    info.energy            = energy;
    info.energyAuxProblems = energy_aux_problems;
    info.minEnergy         = min_energy;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Nested functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function energy_v = cmpEnergy(u1_, u2_, v_)
        sz_u_    = size(u1_);
        sz_v_    = size(v_);
        v_       = scaleImage(v_, [sz_u_, dim_u]);
        v_       = scaleVectorField(v_, sz_u_./sz_v_(1:end-1));
        energy_v = 1/p *  sumAll(abs(u1_ - warpImage(u2_, v_)).^p);
        if(symmetric_OF)
            energy_v = 0.5 * energy_v + 0.5/p *  sumAll(abs(u2_ - warpImage(u1_, -v_)).^p);
        end
        energy_v = energy_v + beta * TVofVelocityField(v_, TV_type, true);
    end

    function v = scaleMotion(v, sizeVNew)
        sz_v_old = size(v);
        v        = scaleImage(v, [sizeVNew, dim_u], wrp_sc_para);
        v        = scaleVectorField(v, sizeVNew./sz_v_old(1:end-1), dim_u+1);
    end

end