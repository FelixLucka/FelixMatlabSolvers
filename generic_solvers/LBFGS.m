function [x, Fx, grad_Fx, x_iterates, info] = LBFGS(FGrad, x, para)
% LBFGS implements a basic version of the Limited-memory
% Broyden-Fletcher-Goldfarb-Shanno optimization algorithm 
%
% DESCRIPTION:
%   The LBFGS algorithm is a quasi-Newton methods that approximates the 
%   Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm using a limited amount 
%   of computer memory.
%   see https://en.wikipedia.org/wiki/Limited-memory_BFGS
%
%  INPUT:
%   FGrad  - function handle that returns objective function and gradient
%            if called as 
%            [F(x), gradF(x)] = FGrad(x)
%   x      - a start value for x
%
%   para  - a struct containing all optional parameters:
%     'memorySize' - number of vectors used for Hessian approximation
%     'initialStepSize' - step size used in first iteration (default:
%     1/norm(gradF(x)
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
%     'displayWarnings' a logical indicating whether warnings should be
%           displayed
%
%     'maxEval' - maximum number of function/gradient evaluations after which to 
%                 stop IN ANY CASE (even if other convergence criteria are not met yet)
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
%   x        - approximate solution to optimization problem
%   Fx       - value of F(x)
%   grad_Fx - value of gradF(x) 
%   x_iterates   - optionally, z(1:returnIteratesInc:end) will be returned
%                 as well
%   info        - information about the iteration
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 16.11.2018
%       last update     - 21.12.2023
%
% See also

clock_cmp_total = tic; % track total computation time

% =========================================================================
% CHECK INPUT AND INITILIZE VARIABLES
% =========================================================================


%%% read out parameters (see above)
memory_size        = checkSetInput(para, 'memorySize', 'i,>0', 10);
[nu, iss_df]       = checkSetInput(para, 'initialStepSize', '>=0', 1);
start_LBFGS_iter  = checkSetInput(para, 'startLBFGSiter', 'i,>0', 1);
stop_criterion     = checkSetInput(para, 'stopCriterion', {'relChangeX', ...
    'absChangeX', 'absGradNorm', 'relGradNorm','maxEval', 'energyIncrease'}, 'maxEval');
stop_tol          = checkSetInput(para, 'stopTolerance', '>=0', 10^-6);
display_warnings  = checkSetInput(para, 'displayWarnings', 'logical', true);
max_eval          = checkSetInput(para, 'maxEval', 'i,>0', inf);
output            = checkSetInput(para, 'output', 'logical', false);
visulization      = checkSetInput(para, 'visulization', 'logical', false);
return_iterates    = checkSetInput(para, 'returnIterates', 'logical', false);
return_iterates_inc = checkSetInput(para, 'returnIteratesInc', 'i,>0', 1);
[errFcn, cmp_err]    = checkSetInput(para, 'errFcn', 'function_handle', 1);
% "soft" upper and lower bounds on x: x will be simply be clipped to them at the end
% of each iteration, IT IS NOT A PROPER L-BFGS-B implementation!
[softBnd, df]     = checkSetInput(para, 'softBnd', 'numeric', [-Inf, Inf]);
apply_soft_bnd    = ~df;
cmp_err = ~cmp_err;



%%% inialize plotting and output
if(visulization)
    visuCommand = checkSetInput(para, 'visuCommand', 'function_handle', 'error');
    visu_para    = checkSetInput(para, 'visuPara', 'struct', emptyStruct);
    visu_para.title = 'starting proximal gradient algorithm';
    [fig_h, axis_h] = visuCommand(x, visu_para);
    visu_para.figureHandle = fig_h;
    visu_para.axisHandle = axis_h;
end

if(output)
    disp('Starting limited memory BFGS optimization')
end


%%% set linesearch parameters
ls_para           = checkSetInput(para, 'lsPara', 'struct', []);
ls_type           = checkSetInput(ls_para, 'type', {'Wolfe', 'backtracking'}, 'Wolfe');
ls_para.maxEval   = checkSetInput(ls_para, 'maxEval', 'i,>0', 20);
ls_para.storeBest = checkSetInput(ls_para, 'storeBest', 'logical', true);
ls_para.tauUp     = checkSetInput(ls_para, 'tauUp',   '>0', 3/2);
ls_para.tauDown   = checkSetInput(ls_para, 'tauDown', '>0', 1/2);
ls_para.c1        = checkSetInput(ls_para, 'c1', '>0', 10^-4);
ls_para.c2        = checkSetInput(ls_para, 'c2', '>0',   0.9);
ls_para.tol       = checkSetInput(ls_para, 'tol', '>0',   0.9);


%%% initialize inner variables
S                   = [];
Y                   = [];
iter                = 0;
FGrad_eval          = 0; 
x_iterates          = {};
sz_x                = size(x);
      
% compute F(x) and gradF(x)
[Fx, grad_Fx]  = FGrad(x);
FGrad_eval     = FGrad_eval + 1;
FGrad_eval_vec = 1;
gradFx_norm    = norm(grad_Fx(:));
energy         = Fx;


%%% compute user specified error function 
if(cmp_err)
   err =  errFcn(x);
end


%%% set scaling factors for resdiuum and norm of x
scale_fac_res = 1;
switch stop_criterion
    case 'relGradNorm'
        stop_value      = norm(grad_Fx(:));
        scale_fac_res   = norm(grad_Fx(:));
    case 'energyIncrease'
        breakCriterion  = stop_criterion;
        stop_value      = stop_tol * 2;
    otherwise
        stop_value      = stop_tol * 2;
end

if(stop_value < stop_tol)
    return 
end

% =========================================================================
% MAIN ITERATION
% =========================================================================


while(FGrad_eval < max_eval)
    
    
    %%% proceed with the iteration
    iter        = iter + 1;
    output_pre  =  ['it ' int2str(iter)];
    output_post = '';
    
    %%% compute search direction 
    if(iter <= start_LBFGS_iter)
        if(iss_df) 
            nu = 1/gradFx_norm;
        end
        update = - grad_Fx;
    else
        update = - twoLoopRecursion(grad_Fx, S, Y);
        if(sum(update .* -grad_Fx) < 0)
            % if update and gradient do not point in the same direction,
            % take normal gradient step (try to increase step size) and
            % check again next iteration
            update = - grad_Fx;
            nu     = 2 * nu_vec(iter - 1);
            start_LBFGS_iter = start_LBFGS_iter + 1;
        end
    end
            
    %%% do linesearch to determine the step
    x_old                = x;
    grad_Fx_old          = grad_Fx;
    ls_para.Fx0          = Fx;
    ls_para.gradFx0      = grad_Fx;
    ls_para.maxEval      = min(ls_para.maxEval, max(1, max_eval - FGrad_eval));
    if(iter == 1)
        % always choose backtracking in first iteration
        ls_para.type       = 'backtracking';
        ls_para.returnGrad = true;
    else
        ls_para.type = ls_type;
    end
    [nu, x, Fx, grad_Fx, add_FGrad_eval] = linesearch(x, update, FGrad, nu, ls_para);
    
        
    %%% update inverse Hessian approximation
    if(iter > memory_size && nu > 0)
        S = [S(:, 2:end), x(:)      - x_old(:)];
        Y = [Y(:, 2:end), grad_Fx(:) - grad_Fx_old(:)];
    else
        S = [S, x(:)      - x_old(:)];
        Y = [Y, grad_Fx(:) - grad_Fx_old(:)];
    end


    %%% apply soft bound constraints
    if(apply_soft_bnd)
        x = max(x, softBnd(1));
        x = min(x, softBnd(2));
    end
    
    
    %%% update tracking variables
    FGrad_eval           = FGrad_eval + add_FGrad_eval;
    FGrad_eval_vec(iter+1) = FGrad_eval;
    gradFx_norm(iter+1)  = norm(grad_Fx(:));
    change_x_norm(iter)  = norm(x(:) - x_old(:));
    energy(iter+1)       = Fx; 
    nu_vec(iter)         = nu;

    
    if(iter == start_LBFGS_iter)
        % set step size to 1 from now on
        nu = 1;
    end
    
    %%% compute user defined error function
    if(cmp_err)
        err(iter+1,:) =  errFcn(x);
    end
    
    
    %%% store iterates (optional)
    if(return_iterates && mod(iter-1,return_iterates_inc) == 0)
        x_iterates{end+1}.x = x;
    end
    
    
    %%% if x does not change, the algorithm won't move anymore
    if(change_x_norm(iter) == 0)
        if(output)
            disp('break: change in x was 0.')
        end
        break
    end
    
    
    %%% check first part of the stop conditions
    switch stop_criterion
        case 'relChangeX'
            stop_value = change_x_norm(iter) / max(norm(x(:)), norm(x_old(:)));
        case 'absChangeX'
            stop_value =  change_x_norm(iter);
        case {'relGradNorm', 'absGradNorm'}
            stop_value = gradFx_norm(iter+1) / scale_fac_res;
    end
    if(stop_value < stop_tol)
        break
    end
    
    
    %%% plotting and output
    output_pre =  [output_pre '; nEvl: ' int2str(FGrad_eval) '/' int2str(max_eval)];
    switch stop_criterion
        case 'maxEval'
            
        otherwise
            output_pre =  [output_pre '; stop value/tol: ' num2str(stop_value/stop_tol,'%.2e')];
    end
    
    output_pre = [output_pre '; en: ' num2str(energy(iter+1), '%.2e')];
    output_pre = [output_pre ' (dFac: ' num2str((energy(iter)-energy(iter+1))/energy(iter+1), '%.2e') ')'];
    output_pre = [output_pre '; relX: ' num2str(change_x_norm(iter)/max(norm(x(:)), norm(x_old(:))), '%.2e')];
    output_pre = [output_pre '; nu: ' num2str(nu_vec(iter), '%.2e')];
    
    if(cmp_err)
        output_pre = [output_pre '; err: ' num2str(err(iter+1,:), '%.4e ')];
    end
    
    output_pre = [output_pre output_post];
    
    if(visulization)
        visu_para.title = output_pre;
        visuCommand(x, visu_para);
    end
    if(output)
        disp(output_pre)
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
    disp(['Limited memory BFGS ended at iteration ' int2str(iter)])
    disp(['Total computation time: ' convertSec(t_cmp_total)]);
end


%%% gather some information
info.iter                = iter;
info.FGradEval           = FGrad_eval;
info.FGradEvalVec        = FGrad_eval_vec;
info.gradientNorm        = gradFx_norm;
info.tCompTotal          = t_cmp_total;
info.energy              = energy;
info.nuVec               = nu_vec;
    
if(cmp_err)
   info.err = err; 
end


end