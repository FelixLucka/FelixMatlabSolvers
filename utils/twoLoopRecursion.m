function q = twoLoopRecursion(q, S, Y, H0)
%FUNCTIONTEMPLATE is a template for a function describtion
%
% DETAILS: 
%
%   "A stochastic L-BFGS approach for full waveform inversion"
%   by Gabriel Fabien-Ouellet, Erwan Gloaguen, Bernard Giroux
%   
%   https://en.wikipedia.org/wiki/Limited-memory_BFGS
%
% USAGE:
%   x = functionTemplate(y)
%
% INPUTS:
%   y - bla bla
%
% OPTIONAL INPUTS:
%   z    - bla bla
%   para - a struct containing further optional parameters:
%       'a' - parameter a
%
% OUTPUTS:
%   x - bla bla
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 05.11.2018
%       last update     - 27.10.2023
%
% See also

% check user defined value for H_0, otherwise assign default value
if(nargin < 4)
    H0 = @(x) x;
end


sz_q  = size(q);
q     = q(:);

n_mem  = size(S, 2);
rho   = 1 ./ sum(S .* Y, 1);
alpha = zeros(n_mem, 1);

for i=n_mem:-1:1
    alpha(i) = rho(i) * sum(S(:,i) .* q);
    q        = q - alpha(i) * Y(:,i);
end

gamma_K = sum(S(:, end) .* Y(:, end)) / sum(Y(:, end).^2);
q       = gamma_K * H0(q);

for i=1:1:n_mem
    beta_i = rho(i) * sum(Y(:, i) .* q);
    q      = q + (alpha(i) - beta_i) * S(:,i);
end

q = reshape(q, sz_q);
end