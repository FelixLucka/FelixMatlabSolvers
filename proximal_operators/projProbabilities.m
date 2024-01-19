function res = projProbabilities(y)
% PROJPROBABILITIES projects the input onto the probability simplex
%
% DESCRIPTION: 
%   projProbabilities.m projects an input vector y onto the probability
%   simplex {x| x_i >= 0, sum(x) = 1} by solving the quadratic problem
%   
%       res = argmin_x 1/2 \| x - y \|_2^2  
%             such that x^T 1 = 1, x >= 0
%
%       by the algorithm described in "Projection onto the probability simplex:
%       An efficient algorithm with a simple proof, and an application"
%       by Wang & Carreira-Perpinan, 2013
%       The code is based on the Matlab code provided there.
%
% USAGE:
%  res = projProbabilities(x)
%
%  INPUT:
%   x - values to project, the projection is performed wrt to the last 
%       dimension 
%
%  OUTPUTS:
%   res - struct containing the results in the form needed by the other
%   algorithms of the toolbox: 
%       'x'  - result of the projection
%       'Jx' - energy of the projection (is set to 0)
%
% ABOUT:
%   author          - Felix Lucka
%   date            - 25.11.2020
%   last update     - 27.10.2023
%
% See also 

if(nDims(y) == 1)
    y = y';
end
sz_y   = size(y);
dim_y  = nDims(y);
d     = sz_y(end);
N     = prod(sz_y(1:end-1));

% we reshape y to a matrix for easier coding (could be avoided)
y     = reshape(y, [], d);
x     = sort(y, 2, 'descend');
aux   = bsxfun(@times, (cumsum(x,2) - 1), 1./(1:d));
x     = bsxfun(@minus, y, aux(sub2ind([N, d], (1:N)', sum(x > aux, 2))));
x     = max(x, 0);
x     = reshape(x, sz_y);

res.x  = x;
res.Jx = 0;

end