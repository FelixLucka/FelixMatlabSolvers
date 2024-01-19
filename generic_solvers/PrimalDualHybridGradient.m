function [x_return, y_return, iter_return, xy_iterates, res_prox_return, info] = ...
    PrimalDualHybridGradient(K, prox_G, prox_Fconj, sigma, tau, theta, x, Kx, y, para)
% PRIMALDUALHYBRIDGRADIENT implements a primal dual hybrid gradient algorithm
% for a specific convex optimization problem.
%
% DESCRIBTION:
% PrimalDualHybridGradient implements a primal dual hybrid gradient algorithm to
% minimize E(x) = G(x) + F(K * x)
% see "A Convex Relaxation Approach for Computing Minimal Partitions" by
% Chambolle, Cremers, Bischof, Pock, 2009, and "A First-Order Primal-Dual Algorithm 
% for Convex Problems with Applications to Imaging" by Chambolle, Pock 2011
% (notation based on the latter)
%
% INPUT:
%   K           - a struct describing K: 
%                 nComponents: number of different components in which K
%                 can be decomposed
%                 fwd: a function handle generating a cell of length
%                 nComponents, each containing K_i * x
%                 adj: a function handle computing sum_i K_i^*  y{i} 
%                 for a cell input y
%   prox_G      - function handle for proximal operator
%                 of G as a function of (z,lambda) defined by
%                 proxG(z,lambda) = argmin_x ( G(x) + 1/(2*lambda) || x - z ||_2^2   )
%                 proxG must return a struct with the field 'x', which is the 
%                 minimizer of the above functional but may contain other fields 
%                 (e.g. for functionals that are defined as J(z) = argmin_w R(z,w), w and J(z) 
%                 can be returned by the Prox
%   prox_Fconj  - function handle for proximal operator
%                 of F* as a function of (z, lambda) defined by
%                 proxFconj(z, lambda) = argmin_x ( F*(x) + 1/(2*lambda) || x - z ||_2^2   )
%                 OR 
%                 of F as a function of (z, lambda) defined by
%                 proxFconj(z, lambda) = argmin_x ( F(x) + 1/(2*lambda) || x - z ||_2^2   )
%                 z must be a cell vector of length nComponents
%                 must return a struct with the field 'x' (as a cell), as above
%   sigma       - primal step size, set to 'auto' to run power iteration to
%                 determine
%   tau         - dual step size, set to 'auto' to run power iteration to
%                 determine
%   theta       - overrelaxation parameter
%   x           - start value for x
%   Kx          - K applied to x
%   y           - start value for y
%
%   para        - a struct containing all optional parameters:
%     proxFnotConj - a logical indicating that proxFconj is the proxy of F and
%                    not of F*
%     'acceleration' - a logical indicating whether an acceleration like
%           the one proposed in Chambolle & Pock, 2011 is used.
%     'returnMinEnergy' - a logical indicating whether an the iterates with
%     the lowest primal energy should be returned, not the last one
%     'uniConvexFunctional' - 'G' or 'F*' to indicate which
%           functional is uniformly convex (both is not implemented yet)
%           ONLY USED IF accelerationFL = true
%     'convexityModulus' - modulus of convexity
%           ONLY USED IF accelerationFL = true
%     'restart' - a logical indicating whether the acceleration should be
%           restarted whenever a rise in the energy is detected
%           ONLY USED IF acceleration = true
%     'maxIter' - maximum number of iteration after which to stop IN ANY CASE
%           (even if other convergence criteria are not met yet)
%
%     'output' - Logical indicating whether output should be
%           displayed
%     'visulization' - Logical indicating whether a visulization of the the
%     single iterates should be performed
%     'monitor' - Logical indicating whether energy should be computed
%     'G' - a function handle to G(x)
%     'F' - a function handle to F(y = Kx), i.e., it is expecting Kx as a
%     cell as input
%     'returnIterates' - a logical indicating whether the iterates should
%           be returned
%     'returnIteratesInc' - if a forth output argument is demanded 1:returnIteratesInc:end iterates will be returned in zIterates
%     'plotCommand' - the command that should be used for plotting (function handle for (x,plotPara))
%     'plotPara' - parameter struct that the plot command takes
%
% OUTPUTS:
%   x_return - primal variable that is returned as result
%   y_return - dual variable that is returned as result (cell)
%   iter_return - number of iterations of the returned result
%   xy_iterates - all x and y (for debugging only)
%   res_prox_returne - the results struct of prox_G at returned result
%   info        - information about the iteration
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 05.04.2018
%       last update     - 27.10.2023
%
% See also ProximalGradientDescend.m

clock_cmp_total = tic; % track total computation time

% =========================================================================
% CHECK INPUT AND INITILIZE VARIABLES
% =========================================================================


%%% read out parameters (see above)
prox_FnotConj       = checkSetInput(para, 'proxFnotConj', 'logical', false);
acceleration        = checkSetInput(para, 'acceleration', 'logical', false);
return_min_energy   = checkSetInput(para, 'returnMinEnergy', 'logical', false);
max_iter            = checkSetInput(para, 'maxIter', 'i,>0', sqrt(numel(x)));
output              = checkSetInput(para, 'output', 'logical', false);
visulization        = checkSetInput(para, 'visulization', 'logical', false);
monitor             = checkSetInput(para, 'monitor', 'logical', true);
return_iterates     = checkSetInput(para, 'returnIterates', 'logical', false);
return_iterates_inc = checkSetInput(para, 'returnIteratesInc', 'i,>0', 1);
lip_power_iter_tol  = checkSetInput(para, 'LipPowerIterTol', '>0', 10^-3);
weight_fac          = checkSetInput(para, 'weightFac', '>0', 1);
compute_energy      = monitor | return_min_energy;


%%% check whether sigma and tau should be pre-computed
if(ischar(sigma) && strcmp(sigma, 'auto'))
    KTK                          = @(x) K.adj(K.fwd(x));
    clock_lip               = tic; % track computation time
    [lipschitz_constant, info_lip] = powerIteration(KTK, size(x), lip_power_iter_tol, 1, output, class(x));
    info.tCompLip          = toc(clock_lip);
    info.LipschitzConstant = lipschitz_constant;
    info.LipInfo           = info_lip;
    sigma                  =  weight_fac * sqrt(0.9 / lipschitz_constant);
    tau                    =  sqrt(0.9 / lipschitz_constant) / weight_fac;
end

%%% 
if(acceleration)
    uni_convex_functional = checkSetInput(para, 'uniConvexFunctional', {'G','F*'}, 'error');
    convexity_modulus     = checkSetInput(para, 'convexityModulus', '>0', 'error');
    restart               = checkSetInput(para, 'restart', 'logical', false);
    if(restart)
        compute_energy = true;
    end
    % safe start values for acceleration
    sigma_0 = sigma;
    tau_0   = tau;
    theta_0 = theta;
end

if(compute_energy)
    G = checkSetInput(para, 'G', 'function_handle', 'error');
    F = checkSetInput(para, 'F', 'function_handle', 'error');
    energy = zeros(max_iter + 1, 1);
end


%%% inialize plotting and output
if(visulization)
    visu_command = checkSetInput(para,'visuCommand','function_handle','error');
    visu_para    = checkSetInput(para,'visuPara','struct',emptyStruct);
    visu_para.title = 'starting primal dual hybrid gradient algorithm';
    [figH,axisH] = visu_command(x,visu_para);
    visu_para.figureHandle = figH;
    visu_para.axisHandle = axisH;
end
if(output)
    disp('Starting primal dual hybrid gradient algorithm')
end


%%% initialize inner variables
Kx_bar          = Kx;
iter            = 0;
iter_return     = 0;
break_iteration = false;
xy_iterates     = {};
if(compute_energy)
    energy(1)  = G(x) + F(Kx);
    min_energy = energy(1);
end
x_return = x;
y_return = y;
res_prox_return = [];

% =========================================================================
% MAIN ITERATION
% =========================================================================

while(~break_iteration)
    
    
    %%% proceed with the iteration
    iter        = iter + 1;
    output_pre  =  ['it ' int2str(iter)];
    output_post = '';
    
    
    %%% update y
    if(prox_FnotConj)
        % proxFconj is the proxy of F, not of F*, therefore, we have to use
        % Moreau's identity to compute the update for y
        proxy_F_res = prox_Fconj(cellfun(@(A,B) A/sigma + B, y, Kx_bar, 'UniformOutput', false), 1 / sigma);
        y           = cellfun(@(A,B,C) A + sigma*B - sigma*C, y, Kx_bar, proxy_F_res.x, 'UniformOutput', false);
    else
        proxy_F_res =  prox_Fconj(cellfun(@(A,B) A + sigma*B,  y, Kx_bar, 'UniformOutput', false), sigma);
        y           = proxy_F_res.x;
    end
    
    
    %%% update x
    x_old       = x;
    Kx_old      = Kx;
    proxy_G_res = prox_G(x - tau * K.adj(y), tau);
    x           = proxy_G_res.x;
    
    
    %%% acceleration
    if(acceleration)
        switch uni_convex_functional
            case 'G'
                theta = 1 / sqrt(1 + 2 * convexity_modulus * tau);
                tau   = theta * tau;
                sigma = sigma / theta;
            case 'F*'
                theta = 1 / sqrt(1 + 2 * convexity_modulus * sigma);
                tau   = tau / theta;
                sigma = theta * sigma;
        end
    end
    
    
    %%% overrelaxation in x (is performed implicitly, because we only need
    %%% K * xBar
    %xBar = x + theta * (x - xOld);
    
    
    %%% go into the next iteration?
    if(iter < max_iter)
        %%% apply K
        Kx    = K.fwd(x);
        Kx_bar = cellfun(@(A,B) A + theta*(A-B), Kx, Kx_old, 'UniformOutput', false);
    else
        break_iteration = true;
    end
    
    
    %%% compute energy of x
    if(compute_energy)
        Gx             = G(x);
        FKx            = F(Kx);
        energy(iter+1) = Gx + FKx;
        min_energy      = min(energy(iter+1), min_energy);
        if(energy(iter+1) <= min_energy)
            % always return the (x,y) with the smallest primal energy
            x_return       = x;
            y_return       = y;
            iter_return    = iter;
            res_prox_return = proxy_G_res;
        elseif(~return_min_energy)
            x_return       = x;
            y_return       = y;
            iter_return    = iter;
            res_prox_return = proxy_G_res;
        end
    else
        x_return       = x;
        y_return       = y;
        iter_return    = iter;
        res_prox_return = proxy_G_res;
    end
    
    %%% restart acceleration?
    if(acceleration && compute_energy && energy(iter+1) > min_energy)
        % restart the acceleration
        if(restart)
            sigma = sigma_0;
            tau   = tau_0;
            theta = theta_0;
            output_post = [output_post, '; *restartAcceleration*'];
        else
            output_post = [output_post, '; !'];
        end
        
    end
    
    
    %%% store iterates (optional)
    if(return_iterates && mod(iter-1,return_iterates_inc))
        xy_iterates{end+1}.x = x;
        xy_iterates{end}.y   = y;
    end
    
    %%% plotting and output
    output_pre = [output_pre '/' int2str(max_iter)];
    if(compute_energy)
        output_pre = [output_pre '; en: ' num2str(energy(iter+1),'%.2e')];
        output_pre = [output_pre ' (dFac: ' num2str((energy(iter)-energy(iter+1))/energy(iter+1), '%.2e') ')'];
    end
    output_pre = [output_pre '; sigma/tau/theta: ' num2str(sigma,'%.2e') '/' num2str(tau,'%.2e') '/' num2str(theta,'%.2e')];
    output_pre = [output_pre output_post];
    if(output)
        disp(output_pre)
    end
    if(visulization)
        visu_para.title = output_pre;
        visu_command(x,visu_para);
    end

    
end

t_cmp_total = toc(clock_cmp_total); 

% =========================================================================
% GATHER RESULTS AND CLEAN UP
% =========================================================================


%%% final output and plotting
if(visulization)
    close(figH);
    drawnow();
end

if(output)
    disp(['Primal dual hybrid gradient algorithm ended at iteration ' ...
    int2str(iter) ', results of iteration ' int2str(iter_return) ' were returned.'])
    disp(['Total computation time: ' convertSec(t_cmp_total)]);
end

%%% gather some information
info.iter       = iter;
info.tCompPDHG  = t_cmp_total;

if(compute_energy)
    info.energy = energy(1:iter+1);
end


end