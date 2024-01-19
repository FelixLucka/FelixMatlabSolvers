function res = proxL1(z, lambda, f, w, lb, ub)
% PROXL1 applies the L1-norm proximal mapping
%
% DESCRIBTION:
%   proxL2 solves the problem
%   min_{ub >= x >= lb} || W (x - f)  ||_1 + 1/(2*lambda) || x - z ||^2_2,
%   where W is a diagonal weighting matrix by a transformation and an 
%   explicit solution via soft-thresholding
%
% INPUT:
%   z      - see above, can be any size
%   lambda - see above, scalar >= 0 (not checked)
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
%       'Jx' -  || W (x - f)  ||_1
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 11.01.2019
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
tau = lambda .* w;

%%% solve min_y  || y  ||_1 + 1/(2*lambdaBar) || y - zBar ||^2_2 via soft thresholding
y  = max(abs(c) - tau, 0) .* sign(c);

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
res.Jx = sum(abs(w(:) .* (res.x(:) - f(:))));

end