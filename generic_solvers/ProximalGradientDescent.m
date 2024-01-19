function [x_return, Fx_return, iter, stop_value, x_iterates, res_proxy_return, info] =...
    ProximalGradientDescent(FGrad, Proxy, nu, x, para)
% PROXIMALGRADIENTDESCENT implements variants of the proximal gradient
% descent algorithm for a general optimization problem
%
% DESCRIPTION:
%   ProximalGradientDescent implements variants of the proximal gradient
%   descent algorithm to solve E(x) = F(x) + J(x)
%   see Chambolle & Pock, An introduction to continuous optimization for 
%   imaging, 2016
%
%  INPUT:
%   FGrad  - function handle that returns objective function and gradient
%            if called as 
%            [F(x), gradF(x)] = FGrad(x)
%   Proxy  - proxmial operator of J as a function handle of x, alpha to solve
%       prox_{J,lambda}(x) = argmin_z ( J(z) + 1/(2*lambda) || z - x ||^2_2)
%       must return a struct with the field 'x', which is the minimizer of the above functional 
%       but may contain other fields (e.g. for functionals that are defined
%       as J(z) = argmin_w R(z,w), w and J(z) can be returned by the Proxy
%   nu     - step size of the gradient step
%   x      - a start value for x
%
%   para  - a struct containing all optional parameters:
%     'J' - function handle for  J. Must take a struct as input (the
%     result of applying the Proxy)
%
%     'inertia'          - inertia parameter (0 = no inertia, default: 0)
%
%     'stepsizeAdaptation' - a logical indicating whether stepsize
%     adaptation should be performed by Barzilai & Borwein / spectral rule
%     (see Goldstein et al., 2015)
%     'adaptationDamping' - a double between 0 and 1 that can be used to
%     dampen the stepsize adaptation, (0 = no damping, 1 = no adaptation)
%
%     'backtrack' - Logical indicating whether a backtracking should be
%     used
%     'backtrackInertia' - logical indicating whether the inertia term
%     should also be backtracked
%     'backtrackMaxIter'  - maximal number of backtracking steps
%     'stepsizeFac'    - a factor < 1 indicating how much the stepsize
%     should be shrunk if the backtracking condition is violated.
%
%     'stopCriterion' - choose ... to stop if ...
%           'relChangeX' if
%               0.5 * norm(x(:)-xOld(:))/(norm(x(:)) + norm(xOld(:))) <  stopTolerance
%           'absChangeX' if
%               0.5 * norm(x(:)-xOld(:))   <  stopTolerance
%           'absGradNorm' if norm(grad(x)) <  stopTolerance
%           'relGradNorm' if norm(grad(x))/norm(grad(x0))
%           'energyIncrease' until the energy increases due to inaccuracies
%           'maxIter' if a maximal number of iterations is reached.
%
%     'stopTolerance' - stop tolerance (see above)
%     'breakCriterion' - break the iteration if
%           'none' never
%           'energyIncrease' when the energy increases
%     'breakTolerance' the tolerance with which a decrease/increase is
%           detected
%     'breakTimes' the number of steps for how long a violation of the
%           breakCriterion is tolerated
%     'displayWarnings' a logical indicating whether warnings should be
%           displayed
%
%     'maxIter' - maximum number of iteration after which to stop IN ANY CASE
%     (even if other convergence criteria are not met yet)
%
%     'output' - Logical indicating whether output should bedisplayed
%     'visulization' - Logical indicating whether a visulization of the the
%     single iterates should be performed
%     'monitor' - Logical indicating whether discrepancy, residuum and
%     norm should be stored (for debugging only)
%     'returnIterates' - a logical indicating whether the iterates should
%                        be returned
%     'returnIteratesInc' - if a forth output argument is demanded 1:returnIteratesInc:end iterates will be returned in zIterates
%     'plotCommand' - the command that should be used for plotting (function handle for (x,plotPara))
%     'plotPara' - parameter struct that the plot command takes
%     'errFnc'   - function that is called on x to compute and plot error
%
%  OUTPUTS:
%   x_return    - approximate solution to optimization problem
%   Fx_return   - value of F(x)
%   iter        - number of iterations used
%   stop_value  - stop value
%   x_iterates  - optionally, z(1:returnIteratesInc:end) will be returned
%                 as well
%   res_proxy_return - returns the all other fields than 'x' (if any) of the results 
%       of applying the Proxy on xReturn (e.g. for functionals that are defined
%       as J(z) = argmin_w R(z,w), w and J(z) can be returned.
%   info        - information about the iteration
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 24.06.2018
%       last update     - 21.12.2023
%
% See also

clock_cmp_total = tic; % track total computation time

% =========================================================================
% CHECK INPUT AND INITILIZE VARIABLES
% =========================================================================


%%% read out parameters (see above)
inertia            = checkSetInput(para, 'inertia', '>=0', 0);
stepsize_adaptation = checkSetInput(para, 'stepsizeAdaptation', 'logical', false);
backtrack           = checkSetInput(para, 'backtrack', 'logical', stepsize_adaptation);
stop_criterion      = checkSetInput(para, 'stopCriterion', {'relChangeX', ...
    'absChangeX', 'absGradNorm', 'relGradNorm','maxIter', 'energyIncrease'}, 'maxIter');
break_criterion   = checkSetInput(para, 'breakCriterion', {'none', 'energyIncrease'}, 'none');
stop_tol          = checkSetInput(para, 'stopTolerance', '>=0', 10^-6);
break_tol         = checkSetInput(para, 'breakTolerance', 'double', 10^-12);
break_times       = checkSetInput(para, 'breakTimes', 'i,>0', 1);
display_warnings  = checkSetInput(para, 'displayWarnings', 'logical', true);
max_iter          = checkSetInput(para, 'maxIter', 'i,>0', sqrt(numel(x)));
output            = checkSetInput(para, 'output', 'logical', false);
visulization      = checkSetInput(para, 'visulization', 'logical', false);
monitor           = checkSetInput(para, 'monitor', 'logical', true);
return_iterates   = checkSetInput(para, 'returnIterates', 'logical', false);
return_iterates_inc = checkSetInput(para, 'returnIteratesInc', 'i,>0', 1);
[errFcn, cmp_err]  = checkSetInput(para, 'errFcn', 'function_handle', 1);
cmp_err = ~cmp_err;

%%% stepsize adaptation parameters
if(stepsize_adaptation)
    % use a damping of the adaptation?
    adaptation_scheme = checkSetInput(para, 'adaptationScheme',...
        {'steepestDes', 'minRes', 'adaptiveBB', 'incBTr'}, 'incBTr');
    switch adaptation_scheme
        case {'steepestDes', 'minRes', 'adaptiveBB'}
            adaptation_damping  = checkSetInput(para, 'adaptationDamping', 'double', 0);
            stepsize_fac_df      = 0.2;
        case 'incBTr'
            adaptation_factor = checkSetInput(para, 'adaptationFactor', 'double', 3/2);
            para.stepsizeAdaptation = true;
            para.stepsizeFac = checkSetInput(para, 'stepsizeFac', 'double', 1/2);
            stepsize_fac_df = 1/2;
    end
    nu_vec = nu;
else
    stepsize_fac_df = 0.9;
end


%%% backtracking parameters
% backtrack interia as well?
backtrack_inertia  = checkSetInput(para, 'backtrackInertia', 'logical', false);
backtrack_max_iter = checkSetInput(para, 'backtrackMaxIter', 'i,>0', 20);
% how much should the stepsize be shrunk?
stepsize_fac       = checkSetInput(para, 'stepsizeFac', '>0', stepsize_fac_df);


%%% inialize plotting and output
if(visulization)
    visu_command = checkSetInput(para, 'visuCommand', 'function_handle', 'error');
    visu_para    = checkSetInput(para, 'visuPara', 'struct', emptyStruct);
    visu_para.title = 'starting proximal gradient algorithm';
    [fig_h, axis_h] = visu_command(x, visu_para);
    visu_para.figureHandle = fig_h;
    visu_para.axisHandle = axis_h;
end

if(output)
    disp('Starting proximal gradient descent')
end


%%% initialize inner variables
x_old               = x;
x_return            = x;
change_x            = x;
res_proxy_return    = [];
inertia_base        = inertia;
iter                = 0;
iter_return         = 0;
FGrad_eval          = 0; 
violation_times     = 0;
x_iterates          = {};
compute_energy      = monitor | strcmp(break_criterion, 'energyIncrease')  |...
                                strcmp(stop_criterion, 'energyIncrease');
proxy_update = emptyStruct;
t_cmp_prox    = 0;
      
% compute F(x) and gradF(x)
[Fx, gradient] = FGrad(x);
FGrad_eval     = FGrad_eval + 1;
gradient_old   = gradient;  
gradient_norm  = norm(gradient(:));


%%% compute energy
if(compute_energy)
    J = checkSetInput(para, 'J', 'function_handle', @(res) res.Jx);
    [Jx, df] = checkSetInput(para, 'Jx', 'double', 0);
    if(df)
        Jx =  J(struct('x', x));
    end
    energy     = Fx + Jx;
    min_energy  = energy;
end


%%% compute user specified error function 
if(cmp_err)
   err =  errFcn(x);
end


%%% set scaling factors for resdiuum and norm of x
scale_fac_res = 1;
switch stop_criterion
    case 'relGradNorm'
        stop_value     = norm(gradient(:));
        scale_fac_res   = norm(gradient(:));
    case 'energyIncrease'
        break_criterion  = stop_criterion;
        stop_value       = stop_tol * 2;
    otherwise
        stop_value       = stop_tol * 2;
end

if(stop_value < stop_tol)
    x_return = x; return % retransform from preconditioning
end

% =========================================================================
% MAIN ITERATION
% =========================================================================


while(iter < max_iter)
    
    
    %%% proceed with the iteration
    iter        = iter + 1;
    output_pre  =  ['it ' int2str(iter)];
    output_post = '';
    
    
    %%% enter backtrack loop 
    backtrack_iter = 0;
    while(1)
        
        %%% compute update of x
        x_new = x - nu * gradient + inertia * (x - x_old);
        % apply proxy
        clock_cmp_prox = tic;
        proxy_res      = Proxy(x_new, nu, proxy_update);
        t_cmp_prox     = t_cmp_prox + toc(clock_cmp_prox);
        x_new          = proxy_res.x;
        
        
        % compute F(x) and gradF(x)
        [Fx_new, gradient_new] = FGrad(x_new);
        FGrad_eval             = FGrad_eval + 1;

        
        if(backtrack && Fx_new > min_energy && backtrack_iter < backtrack_max_iter)
            
            backtrack_iter = backtrack_iter + 1; 
            nu             = stepsize_fac * nu; 
            if(backtrack_inertia)
                inertia    = stepsize_fac * inertia;
            end
            
        else
            
            %%% accept new x and leave backtrack loop
            x_old         =  x;
            gradient_old  = gradient;  
            x        = x_new;
            Fx       = Fx_new;
            gradient = gradient_new;
            change_x  = x - x_old;
            inertia  = inertia_base;
            
            
            if(display_warnings && backtrack_iter == backtrack_max_iter)
                warning('backtracking was not successful')
            end
        
            if(backtrack && backtrack_iter > 0)
                output_post = [output_post '; BTr: ' int2str(backtrack_iter)];
            end
        
            break
            
        end
        
    end
    
    gradient_norm(iter+1) = norm(gradient(:));

    
    %%% compute energy
    if(compute_energy)
        energy(iter+1) = Fx + J(proxy_res);
        min_energy = min(energy);
    end
    
    
    %%% extract proximal operator information here
    if(isfield(proxy_res,'update'))
        proxy_update = proxy_res.update;
        proxy_res    = myRmfield(proxy_res,{'update'});
    end
    
    
    %%% check the break and return conditions
    switch break_criterion
        case 'none'
            violation_times = 0;
        case 'energyIncrease'
            if(energy(iter+1) > min_energy)
                violation_times = violation_times + 1;
                output_post = [output_post, '; ! energy(i) >= min(energy) !'];
            else
                violation_times = 0;
            end
            if(strcmp(stop_criterion, 'energyIncrease'))
                stop_value = energy(iter);
                stop_tol = energy(iter+1);
            end
    end
    break_fl         = violation_times >= break_times;
    return_current_x = violation_times == 0;
        
    if(return_current_x)
        x_return     = x;
        Fx_return    = Fx;
        iter_return = iter;
        res_proxy_return = removeFields(proxy_res,{'x'});
    end
    
    if(break_fl)
        if(output)
            disp(['break criterion ' break_criterion ' was violated!'])
        end
        break
    end
    
    
    %%% if x does not change, the algorithm won't move anymore
    if(max(abs(change_x(:))) == 0)
        if(output)
            disp('break: change in x was 0.')
        end
        break
    end
    
    
    %%% check first part of the stop conditions
    switch stop_criterion
        case 'relChangeX'
            stop_value = norm(change_x(:)) / max(norm(x(:)), norm(x_old(:)));
        case 'absChangeX'
            stop_value =  norm(change_x(:));
        case {'relGradNorm', 'absGradNorm'}
            stop_value = gradient_norm(iter+1) / scale_fac_res;
    end
    if(stop_value < stop_tol)
        break
    end
    
    
    %%% store iterates (optional)
    if(return_iterates && mod(iter-1,return_iterates_inc) == 0)
        x_iterates{end+1}.x = x;
    end
    
    
    %%% compute user defined error function
    if(cmp_err)
        err(iter+1,:) =  errFcn(x);
    end
    
    %%% plotting and output
    switch stop_criterion
        case 'maxIter'
            output_pre =  [output_pre '/' int2str(max_iter)];
        otherwise
            output_pre =  [output_pre '; stop value/tol: ' num2str(stop_value/stop_tol,'%.2e')];
    end
    
    if(compute_energy)
        output_pre = [output_pre '; en: ' num2str(energy(iter+1), '%.2e')];
        output_pre = [output_pre ' (dFac: ' num2str((energy(iter)-energy(iter+1))/energy(iter+1), '%.2e') ')'];
    end
    output_pre = [output_pre '; relX: ' num2str(norm(change_x(:))/max(norm(x(:)), norm(x_old(:))), '%.2e')];
    
    
    output_pre = [output_pre '; nu: ' num2str(nu, '%.2e')];
    
    if(cmp_err)
        output_pre = [output_pre '; err: ' num2str(err(iter+1, :), '%.4e ')];
    end
    
    output_pre = [output_pre output_post];
    
    if(visulization)
        visu_para.title = output_pre;
        visu_command(x, visu_para);
    end
    if(output)
        disp(output_pre)
    end
    
    
    %%% adapt step size nue by hybrid Barzilai & Borwein / spectral rule
    if(stepsize_adaptation)
        switch adaptation_scheme
            case {'steepestDes', 'minRes', 'adaptiveBB'}
                change_grad     = gradient - gradient_old;
                product_changes = change_x(:)' * change_grad(:);
                nu_steep = (change_x(:)' * change_x(:)) / product_changes;  %  steepest descent
                nu_minRes = product_changes / (change_grad(:)' * change_grad(:)); %  minimum residual descent
                nu_minRes = max(nu_minRes, 0);
                switch adaptation_scheme
                    case 'steepestDes'
                        nu_new = nu_steep;
                    case 'minRes'
                        nu_new = nu_minRes;
                    case 'adaptiveBB'
                        % see FASTA guide for this section
                        if(nu_minRes > nu_steep/2)
                            nu_new = nu_minRes;
                        else
                            nu_new = nu_steep - nu_minRes/2;
                        end
                end
                
                if(nu <=0 || isinf(nu) || isnan(nu))
                    % something bad happend, don't change it
                    nu_new = nu;
                    if(display_warnings)
                        warning('something bad happend during the stepsize adaptation, stepsize will only be slightly increased.')
                    end
                end
                % update nu with a damping to avoid cyclic behavoir
                nu = exp( adaptation_damping* log(nu) + (1-adaptation_damping) *  log(nu_new));
            case 'incBTr'
                if(backtrack_iter == 0)
                    nu = nu * adaptation_factor;
                end
        end
        nu_vec(iter+1) = nu;
    end
        
end

t_cmp_total = toc(clock_cmp_total); 

% =========================================================================
% GATHER RESULTS AND CLEAN UP
% =========================================================================


%%% final output and plotting
if(visulization)
    close(fig_h);
    drawnow();
end

if(output)
    disp(['Proximal gradient descent ended at iteration ' int2str(iter) ...
            ', result of iteration ' int2str(iter_return) ' was returned.'])
    disp(['Total computation time: ' convertSec(t_cmp_total)]);
end


%%% gather some information
info.iter                = iter;
info.FGradEval           = FGrad_eval;
info.gradientNorm        = gradient_norm;
info.iterReturn          = iter_return;
info.tCompProxGrad       = t_cmp_total;
info.tCompProx           = t_cmp_prox;

if(compute_energy)
    info.energy = energy;
end
if(stepsize_adaptation)
    info.nuVec = nu_vec(1:iter);
end
if(cmp_err)
   info.err = err; 
end


end