function [x_return, Ax_return, iter, stop_value, x_iterates, res_proxy_return, info] = ...
    ProxGradLinearLeastSquares(A, Atr, b, Sigma, Proxy, nu, x, Ax, Atrb, para)
% PROXGRADLINEARLEASTSQUARES implements variants of the proximal gradient
% algorithm for a regularized linear least-squares problem
%
% DESCRIPTION:
%   ProxGradLinearLeastSquares implements a variants of the proximal gradient
%   algorithm to solve E(x) = 1/2 * || A x - b ||_2^2 + J(x)
%   see Burger, Sawatzky and Steidl, 2014
%   and Tom Goldstein, Christoph Studer and Richard Baraniuk, 2015
%
%  INPUT:
%   A      - function handle for y = A(x)
%   Atr    - function handle for x = A^T(y)
%   b      - the data b
%   Sigma  - positive definite preconditioner
%   Proxy  - proxmial operator as a function handle of x, alpha and Sigma to solve
%       prox_{J,lambda}(x) = argmin_z ( J(z) + 1/(2*lambda) || z - x ||^2_{Sigma^{-1}})
%       must return a struct with the field 'x', which is the minimizer of the above functional 
%       but may contain other fields (e.g. for functionals that are defined
%       as J(z) = argmin_w R(z,w), w and J(z) can be returned by the Proxy
%   nu     - step size of the gradient step
%   x      - a start value for x
%   Ax     - A applied to that start value
%   Atrb   - A^T applied to b
%
%   para  - a struct containing all optional parameters:
%     'J' - function handle for  J. Must take a struct as input (the
%     result of applying the Proxy)
%     'stepsizeAdaptation' - a logical indicating whether stepsize
%     adaptation should be performed by Barzilai & Borwein / spectral rule
%     (see Goldstein et al., 2015)
%     'adaptationDamping' - a double between 0 and 1 that can be used to
%     dampen the stepsize adaptation, (0 = no damping, 1 = no adaptation)
%
%     'backtrack' - Logical indicating whether a backtracking should be
%     used
%     'monotonBacktrackingFL - Logical indicating whether monotonic or non-monotonic
%     backtracking should be used
%     'backtrackM'  - integer denoting the number of former values of
%     1/2 * || A x - b ||_2^2 to be considered in backtracking.
%     'backtrackMaxIter'  - maximal number of backtracking steps
%     'stepsizeFac'    - a factor < 1 indicating how much the stepsize
%     should be shrunk if the backtracking condition is violated.
%
%     'fastGradient' - Logical indicating whether a two-step update
%     scheme should be used
%     'restart' should the gradient acceleration be restarted when the
%     enery increases?
%     'monotonFastGradient' - Logical indicating whether monosticity of the objective
%     function is enforced in the two-step scheme
%     'tauUpdateRule' - choose 'FISTA' or 'Nesterov' for two different
%     two-step schemes
%
%     'stopCriterion' - choose ... to stop if ...
%           'absDis' if  || A x - b || < stopTolerance
%           'relRes' if  || A x - b ||/||f|| <  stopTolerance
%           'absRes': the same as 'absDis'
%           'relChangeX' if
%               0.5 * norm(x(:)-xOld(:))/(norm(x(:)) + norm(xOld(:))) <  stopTolerance
%           'absChangeX' if
%               0.5 * norm(x(:)-xOld(:)) <  stopTolerance
%           'normResOptCon','ratioResOptCon','hybrResOptCon': stoping
%           criteria based on the residuum of the optimality condition, see
%           Goldstein et al., 2015
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
%
%  OUTPUTS:
%   xReturn     - approximate solution to optimization problem
%   FxReturn    - value of F(x)
%   iter        - number of iterations used
%   stopValue   - stop value
%   xIterates   - optionally, z(1:returnIteratesInc:end) will be returned
%                 as well
%   resProxyReturn - returns the all other fields than 'x' (if any) of the results 
%       of applying the Proxy on xReturn (e.g. for functionals that are defined
%       as J(z) = argmin_w R(z,w), w and J(z) can be returned.
%   info        - information about the iteration
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 05.04.2018
%       last update     - 21.12.2023
%
% See also

clock_cmp_total = tic; % track total computation time

% =========================================================================
% CHECK INPUT AND INITIALIZE VARIABLES
% =========================================================================

% Preconditioner
if(nargin < 4 || isempty(Sigma))
    Sigma = @(x) x;
end

% Proxy
if(nargin < 5 || isempty(Proxy))
    Proxy = @(x, nu, Sigma, proxy_update) struct('x', x, 'Jx', 0);
end

% nu
if(nargin < 6 || isempty(nu))
    nu = 1;
end

% Atrb
if(nargin < 9 || isempty(Atrb))
    Atrb = Atr(b);
end

% x
if(nargin < 7 || isempty(x))
    x = zeros(size(Atrb), 'like', Atrb);
end

% Ax
if(nargin < 8 || isempty(Ax))
    Ax = A(x);
end

info = [];


%%% read out parameters (see above)
stepsize_adaptation = checkSetInput(para, 'stepsizeAdaptation', 'logical', false);
fast_gradient       = checkSetInput(para, 'fastGradient', 'logical', not(stepsize_adaptation));
backtrack           = checkSetInput(para, 'backtrack', 'logical', stepsize_adaptation);
stop_criterion      = checkSetInput(para, 'stopCriterion', {'absDis', 'relRes', ...
    'absRes', 'relChangeX', 'absChangeX', 'normResOptCon', 'ratioResOptCon',...
    'hybrResOptCon', 'maxIter', 'energyIncrease'}, 'maxIter');
break_criterion   = checkSetInput(para, 'breakCriterion', {'none', 'energyIncrease'}, 'none');
stop_tol          = checkSetInput(para, 'stopTolerance', '>=0', 10^-6);
break_tol         = checkSetInput(para, 'breakTolerance', 'double', 10^-12);
break_times       = checkSetInput(para, 'breakTimes', 'i,>0', 1);
display_warnings  = checkSetInput(para, 'displayWarnings', 'logical', true);
max_iter          = checkSetInput(para, 'maxIter', 'i,>0', sqrt(numel(x)));
output            = checkSetInput(para, 'output', 'logical', false);
visulization      = checkSetInput(para, 'visulization', 'logical', false);
monitor           = checkSetInput(para, 'monitor', 'logical', true);
return_iterates    = checkSetInput(para, 'returnIterates', 'logical', false);
return_iterates_inc = checkSetInput(para, 'returnIteratesInc', 'i,>0', 1);


%%% stepsize adaptation parameters
if(stepsize_adaptation)
    % use a damping of the adaptation?
    adaptation_scheme = checkSetInput(para, 'adaptationScheme',...
        {'steepestDes', 'minRes', 'adaptiveBB', 'incBTr'}, 'adaptiveBB');
    switch adaptation_scheme
        case {'steepestDes', 'minRes', 'adaptiveBB'}
            adaptation_damping  = checkSetInput(para, 'adaptationDamping', 'double', 0);
            stepsize_fac_df     = 0.2;
        case 'incBTr'
            adaptation_fac = checkSetInput(para, 'adaptationFactor', 'double', 3/2);
            para.stepsizeAdaptation = true;
            para.stepsizeFac = checkSetInput(para, 'stepsizeFac', 'double', 1/3);
            stepsize_fac_df = 1/3;
    end
    nu_vec(1) = nu;
else
    stepsize_fac_df = 0.9;
end


%%% gradient acceleration parameters
if(fast_gradient)
    tau_update_rule  = checkSetInput(para, 'tauUpdateRule', {'FISTA', 'Nesterov'}, 'FISTA');
    t                    = 1;
    restart              = checkSetInput(para, 'restart', 'logical', true);
    stepsize_fac_restart_df = checkSetInput(para, 'stepsizeFacRestartDF', '>0', 0.5);
    backtrack_max_iter   = checkSetInput(para, 'backtrackMaxIter', 'i,>0', 3);
elseif(display_warnings && fast_gradient && stepsize_adaptation)
    warning('Combination of stepsize adaptation and gradient acceleration not recommended.')
end


%%% backtracking parameters
if(backtrack)
    % use monotonic backtracking?
    monoton_backtracking = checkSetInput(para, 'monotonBacktracking', 'logical', true);
    if(~monoton_backtracking)
        % maximum over how many past steps to search for?
        backtracking_m = checkSetInput(para, 'backtrackingM', 'i,>0', 10);
    end
    backtrack_max_iter = checkSetInput(para, 'backtrackMaxIter', 'i,>0', 20);
    % how much should the stepsize be shrunk?
    stepsize_fac      = checkSetInput(para, 'stepsizeFac', '>0', stepsize_fac_df);
end


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
    disp('Starting proximal gradient algorithm')
end


%%% initialize inner variables
x_old                  = x;
x_return               = x;
change_x               = x;
res_opt_con            = 0;
res_opt_con_normalized = 0;
Ax_return              = Ax;
residuum               = Ax - b;
res_proxy_return       = [];
iter_return            = 0;
residuum_norm          = norm(residuum(:));
data_fidelty           = 0.5*residuum_norm^2;
iter                   = 0;
iter_return            = 0;
violation_times        = 0;
x_iterates             = {};
compute_energy         = monitor | strcmp(break_criterion, 'energyIncrease')  |...
                                strcmp(stop_criterion, 'energyIncrease');


%%% set scaling factors for resdiuum and norm of x
scale_fac_res = 1;
switch stop_criterion
    case 'relRes'
        scale_fac_res    = checkSetInput(para, 'bNorm', '>0', 'error');
        stop_value       = residuum_norm/scale_fac_res;
    case {'absRes', 'absDis'}
        stop_value       = residuum_norm;
    case 'energyIncrease'
        break_criterion  = stop_criterion;
        stop_value       = stop_tol * 2;
    otherwise
        stop_value       = stop_tol * 2;
end

if(stop_value < stop_tol)
    x_return = x; return % retransform from preconditioning
end


%%% compute energy
if(compute_energy)
    J = checkSetInput(para, 'J', 'function_handle', @(res) res.Jx);
    [Jx, df] = checkSetInput(para, 'Jx', 'double', 0);
    if(df)
        Jx =  J(struct('x', x));
    end
    energy     = 0.5 * norm(residuum(:))^2 + Jx;
    min_energy = energy;
end


%%% compute A^T * A * x and build gradient
if(any(Ax(:)))
    AtrAx   = Atr(Ax);
else
    AtrAx   = zeros(size(x));
end
AtrAx_old    = AtrAx;
gradient     = (AtrAx - Atrb);
proxy_update = emptyStruct;
t_cmp_prox   = 0;


% =========================================================================
% MAIN ITERATION
% =========================================================================


while(iter < max_iter)
    
    
    %%% proceed with the iteration
    iter        = iter + 1;
    output_pre  =  ['it ' int2str(iter)];
    output_post = '';
    
    
    %%% compute decend direction and do forward step
    if(fast_gradient)
        switch tau_update_rule
            case 'Nesterov'
                tau = (t - 1) / (t + 2); % see Burger, Sawatzky and Steidl, 2014
                t 	= t + 1;
            case 'FISTA'
                t_old = t;
                t     = (1 + sqrt(1 + 4*t_old^2)) / 2;
                tau   = (t_old - 1) / t;
        end
        % store gradient before acceleration
        descent_direction = Sigma(gradient - tau/nu * (x - x_old) + tau * (AtrAx - AtrAx_old));
    else
        % just apply preconditioning
        descent_direction = Sigma(gradient);
    end
    % forward step
    x_update = x - nu * descent_direction;
    
    
    %%% apply proximity operator
    x_old           =  x;
    clock_cmp_prox  = tic; % track time to apply proxy
    proxy_res       = Proxy(x_update, nu, Sigma, proxy_update);
    t_cmp_prox      = t_cmp_prox + toc(clock_cmp_prox);
    x               = proxy_res.x;
    change_x        = x - x_old;
    
    
    %%% compute Ax
    if(any(x(:)))
        Ax    =  A(x);
    else
        Ax = zeros(size(Ax));
    end
    residuum             = Ax - b;
    residuum_norm(iter+1) = norm(residuum(:));
    data_fidelty(iter+1)  = 0.5 * residuum_norm(iter+1)^2;
    
    
    %%% compute energy
    if(compute_energy)
        energy(iter+1) = data_fidelty(iter+1) + J(proxy_res);
        min_energy = min(energy);
    end
    
    
    %%% backtracking line search
    if(backtrack)
        
        if(monoton_backtracking)
            data_fidelty_past_max = data_fidelty(iter); % monotonic line search
        else
            data_fidelty_past_max = max(data_fidelty(max(end-backtracking_m-1,1):1:end-1));
        end
        
        backtrack_iter = 0;
        backtrack_con = data_fidelty(iter+1) - 1e-12 > ...
            data_fidelty_past_max + change_x(:)'*descent_direction(:) + norm(change_x(:))^2/(2*nu);
        %         if(computeEnergyFL) % ensure energy decrease
        %             backtrackCon = backtrackCon | energy(iter+1) > minEnergy;
        %         end
        
        % perform backtracking (repeats above steps)
        while backtrack_con && backtrack_iter < backtrack_max_iter
            nu            = stepsize_fac * nu;
            x_update       = x_old - nu*descent_direction;
            clock_cmp_prox = tic;
            proxy_res      = Proxy(x_update, nu, Sigma, proxy_update);
            t_cmp_prox     = t_cmp_prox + toc(clock_cmp_prox);
            x             = proxy_res.x;
            change_x       = x - x_old;
            if(any(x(:)))
                Ax    =  A(x);
            else
                Ax    = zeros(size(Ax));
            end
            residuum             = Ax - b;
            residuum_norm(iter+1) = norm(residuum(:));
            data_fidelty(iter+1)  = 0.5*residuum_norm(iter+1)^2;
            backtrack_iter        = backtrack_iter + 1;
            backtrack_con = data_fidelty(iter+1) - 1e-12 > ...
                data_fidelty_past_max + change_x(:)'*descent_direction(:) + norm(change_x(:))^2/(2*nu);
            
            % compute energy
            if(compute_energy)
                energy(iter+1) = data_fidelty(iter+1) + J(proxy_res);
                min_energy      = min(energy);
                % backtrackCon = backtrackCon | energy(iter+1) > minEnergy;
            end
        end
        
        if(display_warnings && backtrack_con)
            warning('backtracking was not successful')
        end
        
        output_post = [output_post '; BTr: ' int2str(backtrack_iter)];
        
    end
    
    if(fast_gradient && compute_energy && energy(iter+1) > min_energy)
        % if we use the fast gradient approach without backtracking, we
        % switch to the normal gradient whenever the energy would increase
        
        % restart the gradient acceleration (next loop will start with a normal gradient step)
        if(restart)
            t = 1;
        else
            % switch off the fastGradientFL
            fast_gradient = false; % stop fast gradient stuff
            if(display_warnings)
                warning('fast gradient acceleration was switched off.')
            end
        end
        output_post = [output_post, '; *restartAcceleration*'];
        
        % perform backtracking until energy decreases in gradient direction
        backtrack_iter = 0;
        backtrack_con  = true;
        while backtrack_con && backtrack_iter < backtrack_max_iter
            descent_direction = Sigma(gradient);
            x_update       = x_old - stepsize_fac_restart_df^backtrack_iter * nu*descent_direction;
            clock_cmp_prox = tic;
            proxy_res = Proxy(x_update, stepsize_fac_restart_df^backtrack_iter * nu, Sigma, proxy_update);
            t_cmp_prox     = t_cmp_prox + toc(clock_cmp_prox);
            x              = proxy_res.x;
            change_x       = x - x_old;
            if(any(x(:)))
                Ax =  A(x);
            else
                Ax = zeros(size(Ax));
            end
            residuum              = Ax - b;
            residuum_norm(iter+1) = norm(residuum(:));
            data_fidelty(iter+1)  = 0.5 * residuum_norm(iter+1)^2;
            energy(iter+1)        = data_fidelty(iter+1) + J(proxy_res);
            min_energy            = min(energy);
            
            backtrack_iter = backtrack_iter + 1;
            backtrack_con  = energy(iter+1) > min_energy && ~strcmp(break_criterion,'energyIncrease');
        end
        
        if(display_warnings && fast_gradient && backtrack_con && ~strcmp(break_criterion,'energyIncrease'))
            warning('switching to normal instead of fast gradient and backtracking did not prevent an increase in energy functional!')
        end
        if(backtrack_iter > 1)
            output_post = [output_post, '; BTr:' int2str(backtrack_iter-1)];
        end
        
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
        x_return = x;
        Ax_return    = Ax;
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
        case {'absRes', 'relRes', 'absDis'}
            stop_value = residuum_norm(iter+1) / scale_fac_res;
    end
    if(stop_value < stop_tol)
        break
    end
    
    
    %%% store iterates (optional)
    if(return_iterates && mod(iter-1,return_iterates_inc) == 0)
        x_iterates{end+1}.x = x;
    end
    
    
    if(iter < max_iter)
        
        %%% compute AtrAx and build gradient
        AtrAx_old    = AtrAx;
        if(any(Ax(:)))
            AtrAx   = Atr(Ax);
        else
            AtrAx   = zeros(size(x));
        end
        gradient_old = gradient;
        gradient    = (AtrAx - Atrb);
        
        % estimate the residual of the optimality condtion
        res_opt_con(iter)           = gather(norm(gradient(:) + (x_update(:) -  x(:))/nu));
        res_opt_con_normalized(iter) = res_opt_con(iter) / ...
            gather(max(norm(gradient(:)), norm(x(:) - x_update(:))/nu) + stop_tol/100);

        
        %%% check second part of the stop conditions
        switch stop_criterion
            case 'normResOptCon'
                stop_value = res_opt_con_normalized(iter);
            case 'ratioResOptCon'
                stop_value = res_opt_con(iter) / (max(res_opt_con) + stop_tol/100);
            case 'hybrResOptCon'
                stop_value = min(res_opt_con(iter) / (max(res_opt_con) + stop_tol/100), res_opt_con_normalized(iter));
        end
        if(stop_value < stop_tol)
            break
        end
        
        
        %%% plotting and output
        switch stop_criterion
            case 'maxIter'
                output_pre =  [output_pre '/' int2str(max_iter)];
            otherwise
                output_pre =  [output_pre '; stop val/tol: ' num2str(stop_value/stop_tol,'%.2e')];
        end
        
        if(compute_energy)
            output_pre = [output_pre '; en: ' num2str(energy(iter+1), '%.2e')];
            output_pre = [output_pre ' (dFac: ' num2str((energy(iter)-energy(iter+1))/energy(iter+1), '%.2e') ')'];
        end
        output_pre = [output_pre '; relX: ' num2str(norm(change_x(:))/max(norm(x(:)), norm(x_old(:))), '%.2e')];

                        
        output_pre = [output_pre '; nu: ' num2str(nu, '%.2e')];
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
                            nuNew = nu_steep;
                        case 'minRes'
                            nuNew = nu_minRes;
                        case 'adaptiveBB'
                            % see FASTA guide for this section
                            if(nu_minRes > nu_steep/2)
                                nuNew = nu_minRes;
                            else
                                nuNew = nu_steep - nu_minRes/2;
                            end
                    end
                    
                    if(nu <=0 || isinf(nu) || isnan(nu))
                        % something bad happend, don't change it
                        nuNew = nu;
                        if(display_warnings)
                        warning('something bad happend during the stepsize adaptation, stepsize will only be slightly increased.')
                        end
                    end
                    % update nu with a damping to avoid cyclic behavoir
                    nu = exp( adaptation_damping* log(nu) + (1-adaptation_damping) *  log(nuNew));
                case 'incBTr'
                    nu = nu * adaptation_fac;
            end
            nu_vec(iter+1) = nu;
        end
        
        
    else % (iter == maxIter)
        
        
        %%% last iter, diplay output
        switch stop_criterion
            case 'maxIter'
                output_pre =  [output_pre '/' int2str(max_iter)];
            otherwise
                output_pre =  [output_pre '; stop value / stop tolerance  was not computed anymore'];
        end
        
        if(compute_energy)
            output_pre = [output_pre '; en: ' num2str(energy(iter+1), '%.2e')];
            output_pre = [output_pre ' (dFac: ' num2str((energy(iter)-energy(iter+1))/energy(iter+1), '%.2e') ')'];
        end
        output_pre = [output_pre '; nu: ' num2str(nu, '%.2e')];
        output_pre = [output_pre output_post];
        
        if(output)
            disp(output_pre)
        end
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
    disp(['Proximal gradient algorithm ended at iteration ' int2str(iter) ...
            ', result of iteration ' int2str(iter_return) ' was returned.'])
    disp(['Total computation time: ' convertSec(t_cmp_total)]);
end


%%% gather some information
info.iter                = iter;
info.dataFidelty         = data_fidelty;
info.residuumNorm        = residuum_norm;
info.resOptCon           = res_opt_con;
info.resOptConNormalized = res_opt_con_normalized;
info.iterReturn          = iter_return;
info.tCompProxGrad       = t_cmp_total;
info.tCompProx           = t_cmp_prox;

if(compute_energy)
    info.energy = energy;
end
if(stepsize_adaptation)
    info.nuVec = nu_vec;
end


end