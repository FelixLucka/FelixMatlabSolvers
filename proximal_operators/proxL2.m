function res = proxL2(z, lambda, f, w, lb, ub)
% PROXL2 applies the L2 proximal mapping
%
% DESCRIBTION:
%   proxL2 solves the problem
%   min_{ub >= x >= lb} 1/2 || W (x - f)  ||^2_2 + 1/(2*lambda) || x - z ||^2_2,
%   where W is a diagonal weighting matrix
%
% INPUT:
%   z     - see above, can be any size
%   alpha - see above, scalar >= 0 (not checked)
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
%       'Jx' - 1/2 || W (x - f)  ||^2_2
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 05.04.2018
%       last update     - 27.10.2023
%
% See also prox*.m

% check user defined value for f, otherwise assign default value
if(nargin < 3)
    f = 0;   
end

% check user defined value for w, otherwise assign default value
if(nargin < 4)
    w = 1;   
end


%%% solve quadratic problem
res.x  = (z + lambda .* w.^2 .* f) ./ (1 + lambda .* w.^2);


%%% apply constraints

% if lower bound is specified, apply
if(nargin > 4)
    res.x  = max(lb, res.x); 
end

% if lower bound is specified, apply
if(nargin > 5)
    res.x  = min(ub, res.x); 
end

%%% compute energy
res.Jx = 0.5 * sum((w(:) .* (res.x(:) - f(:))).^2);

end