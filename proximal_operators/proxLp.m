function res = proxLp(z, lambda, p, f, w, lb, ub)
% PROXLp applies the Lp-norm proximal mapping
%
% DESCRIBTION:
%   proxL2 solves the problem
%   min_{ub >= x >= lb} 1/p || W (x - f)  ||^p_p + 1/(2*lambda) || x - z ||^2_2,
%   where W is a diagonal weighting matrix. To solve it, the problem is
%   first transformed into 
%   min_y 1/p || y  ||^p_p + 1/(2*tau) || y - c ||^2_2,
%   for p = 1, 2, there are explicit solutions. For 1 < p < 2, 
%   the solution is given by y = sign(c) q, where q solves 
%   q + p tau q^(p-1) = abs(c)
%   (see "Proximal Mapping Methods in Signal Processing" by Combettes and 
%   Pesquet). We use a Newton's method to solve this 1D non-linear equation
%   to machine precision.
%
% INPUT:
%   z      - see above, can be any size
%   lambda - see above, field >= 0 (not checked)
%   p      - see above, 2 >= p >= 1 
% 
% OPTIONAL INPUT
%   f     - see above, same size as z or autoexpand 
%   w     - see above, same size as z or autoexpand 
%   lb    - lower bound on x
%   ub    - upper bound on x
%
%  OUTPUTS:
%   res - struct containing the results in the form needed by the other
%   algorithms of the toolbox: 
%       'x'  - solution of the above problem
%       'Jx' - 1/p || W (x - f)  ||^p_p
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 23.12.2018
%       last update     - 27.10.2023
%
% See also prox*.m

% check user defined value for f, otherwise assign default value
if(nargin < 4)
    f = 0;   
end

% check user defined value for w, otherwise assign default value
if(nargin < 5)
    w = 1;   
end


%%% set up affine linear tranformation
c   = w .* (z-f);
tau = bsxfun(@times, w.^p, lambda);

%%% solve min_y 1/p || y  ||^p_p + 1/(2*lambdaBar) || y - zBar ||^2_2
if(p == 1)
    
    % explicit solution via soft thresholding
    y  = max(abs(c) - tau, 0) .* sign(c);
    
elseif(p == 2)
    
    % explicit solution via optimality condition
    y  = c ./ (1 + tau);
    
elseif(1 < p && p < 2)
   
    %%% Newton scheme to solve f(q) = q + p tau q^(p-1) - abs(c) = 0
    
    % start from solution to p = 1 problem if it is > 0
    q    = max(abs(c) - tau, 0);
    % for the rest, start from p = 2 solution divided by 10 so many times
    % that f(q) is negative 
    ready     = q > 0;
    q(~ready) = abs(c(~ready)) ./ (1 + tau);
    while not(all(ready(:)))
        q(~ready)       = q(~ready)/10;
        fun_val         = q(~ready) + p * tau * q(~ready).^(p-1) - abs(c(~ready));
        ready((~ready)) = fun_val <= 0;
    end
    
    % Newton scheme. We start to the left of the root. Due to the monotone 
    % decay of the first derivative, the scheme never overshoots in theory.
    % Practically, we stop as soon as funVal >= 0
    ready = false(size(q));
    while not(all(ready(:)))
        fun_val        = q(~ready) + p * tau * q(~ready).^(p-1) - abs(c(~ready));
        derivative_val = 1 + (p-1) * p * tau * q(~ready).^(p-2);
        q(~ready)      = max(0, q(~ready) - fun_val./derivative_val);
        ready(~ready)  = fun_val >= 0;
    end
    y = sign(c) .* q;
    
else
   error('invalid value for p, choose p in [1,2]') 
end

    
%%% reverse affine linear tranformation
res.x = y ./ w + f;
    
%%% apply constraints

% if lower bound is specified, apply
if(nargin > 5)
    res.x  = max(lb, res.x); 
end

% if lower bound is specified, apply
if(nargin > 6)
    res.x  = min(ub, res.x); 
end

%%% compute energy
res.Jx = sum(abs(w(:) .* (res.x(:) - f(:))).^p) / p;

end