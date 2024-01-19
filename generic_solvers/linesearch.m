function [nu, x, Fx, grad_Fx, n_eval] = linesearch(x0, p, F_grad, nu, para)
%LINESEARCH IMPLEMENTS DIFFERENT LINESEARCH STRATEGIES
%
% DETAILS:
%   linesearch.m implements different linesearch strategies, including 
%       backtracking: 
%       Wolfe:
%       fminbnd: 
%   see ???
%
% USAGE:
%   [nu, x, Fx, grad_Fx, n_eval] = linesearch(x0, p, F_grad, nu, para)
%
% INPUTS:
%   x0     - starting point
%   p      - search direction
%   FGrad  - function handle that returns objective function and gradient
%            if called as
%            [F(x), gradF(x)] = FGrad(x)
%   nu     - initial step size
%   para - a struct containing further optional parameters:
%       type - see above: 'backtracking', 'Wolfe' or 'fminbnd'
%       tauUp - factor by which to increase step size (df: 3/2)
%       tauDown - factor by which to decrease step size (df: 1/2)
%       maxEval - maximal number of evaluation of FGrad (df: 10^3)
%       maxNuChange - maximal change of step size (df: 10^12)
%       storeBest - boolean indicating whether the step leading to the best
%                   function value should be stored and returned (df:
%                   false)
%
% OUTPUTS:
%   nu - chosen step size
%   x  - x0 + nu * p
%   Fx - F(x) 
%   grad_Fx - gradient of F at x
%   n_eval - number of evaluation of FGrad
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 05.11.2018
%       last update     - 21.12.2023
%
% See also

nu0         = nu;

type          = checkSetInput(para, 'type', {'backtracking', 'Wolfe', 'fminbnd'}, 'fminbnd');
tau_up        = checkSetInput(para, 'tauUp',   '>0', 3/2);
tau_down      = checkSetInput(para, 'tauDown', '>0', 1/2);
max_eval      = checkSetInput(para, 'maxEval', 'i,>0', 10^3);
max_nu_change = checkSetInput(para, 'maxNuChange', '>0', 10^12);
store_best    = checkSetInput(para, 'storeBest', 'logical', false);

switch type
    case 'backtracking'
        [Fx0, df]   = checkSetInput(para, 'Fx0', 'numeric', 0);
        return_grad = checkSetInput(para, 'returnGrad', 'logical', false);
        cmp_grad    = checkSetInput(para, 'cmpGrad', 'logical', false);
        grad_Fx0    = []; 
        if(df)
            Fx0 = F_grad(x0);
        end
    case 'Wolfe'
        Fx0      = checkSetInput(para, 'Fx0',     'numeric', 'error');
        grad_Fx0 = checkSetInput(para, 'gradFx0', 'numeric', 'error');
        c1       = checkSetInput(para, 'c1', '>0', 10^-4);
        c2       = checkSetInput(para, 'c2', '>0',   0.9);
        cmp_grad = true;
    case 'fminbnd'
        [Fx0, df]  = checkSetInput(para, 'Fx0', 'numeric', 0);
        return_grad = checkSetInput(para, 'returnGrad', 'logical', false);
        grad_Fx0   = []; 
        if(df)
            Fx0 = F_grad(x0);
        end
        cmp_grad = false;
        nu_vec          = 0;
        Fx_vec          = Fx0;
end
grad_Fx = [];
Fx     = Fx0;

if(store_best)
   nu_best      = 0;
   x_best       = x0;
   Fx_best      = Fx0;
   grad_Fx_best = grad_Fx0;
end

%%% first bloc of operations
n_eval = 0;
while(1)
    
    % take one step
    Fx_old       = Fx;
    x            = x0 + nu * p;
    if(cmp_grad)
        % compute F(x) and grad[F](x)
        [Fx, grad_Fx] = F_grad(x);
    else
        % compute F(x)
        Fx = F_grad(x);
    end
    n_eval        = n_eval + 1;
    
    if(store_best && Fx < Fx_best)
        nu_best     = nu;
        x_best      = x;
        Fx_best     = Fx;
        grad_Fx_best = grad_Fx;
    end
    
    switch type
        case 'backtracking'
            
            if(Fx < Fx0)
               break
            else
                nu = nu * tau_down;
            end
            
        case 'Wolfe'
            
            p_grad_F       = sum(p(:) .* grad_Fx0(:));
            % check Armijo rule and curvature condition
            Armijo_rule     = Fx <= Fx0 + c1 * nu * p_grad_F;
            curvature_cond  = - sum(p(:) .* grad_Fx(:)) <= - c2 * p_grad_F;
            
            if((Armijo_rule && curvature_cond))
                break
            elseif(~Armijo_rule)
                nu = nu * tau_down;
            elseif(~curvature_cond)
                nu = nu * tau_up;
            else
                error('linesearch error, neither Armijo rule nor curvature condition fulfilled!')
            end
            
        % the following methods use this loop to find a starting interval
        % for a refined search
        case 'fminbnd'
            nu_vec(end+1) = nu;
            Fx_vec(end+1) = Fx;
            if(Fx_vec(end) < Fx_vec(end-1))
                % we still need to find an intervall by enlarging step size
                nu = nu * tau_up;
            else
                % define left and right boundaries of interval
                ind    = find(diff(Fx_vec) >= 0, 1, 'first');
                nu_int  = nu_vec([ind, ind + 1]);
                break
            end
    end
    
    if(n_eval >= max_eval)
        break
    end
    
    if(max(nu/nu0, nu0/nu) > max_nu_change)
        error('linesearch error, step size changed too much!')
    end
end


%%% second bloc of operations

switch type
    case 'backtracking'
        % nothing to be done
    case 'Wolfe'
        % nothing to be done
    case 'fminbnd'
        if(n_eval < max_eval)
            % prepare call to fminbnd
            funH = @(nu) F_grad(x0 + nu * p);
            options = optimset('fminbnd');
            options.MaxFunEvals = max_eval - n_eval;
            options.TolX        = eps(nu)/10;
            options.Display     = 'off';
            [nu,Fx,exitflag,outputFminbnd] = fminbnd(funH, nu_int(1), nu_int(2), options);
            x = x0 + nu * p;
            
            if(store_best && Fx < Fx_best)
                nu_best     = nu;
                x_best      = x;
                Fx_best     = Fx;
                grad_Fx_best = grad_Fx;
            end
        end
end

    
if(store_best)
    nu     = nu_best;
    x      = x_best;
    Fx     = Fx_best;
    grad_Fx = grad_Fx_best;
end

if(isempty(grad_Fx) && return_grad)
    % compute grad[F](x)
    [~, grad_Fx] = F_grad(x);
end

end