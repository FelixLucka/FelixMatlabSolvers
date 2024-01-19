function [Fx, gradFx]  = FGradLinearLeastSquares(Gmat, x, y, theta, stochasticity, rngSeed)
%FUNCTIONTEMPLATE is a template for a function describtion
%
% DETAILS: 
%   functionTemplate.m can be used as a template 
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
%       last update     - 05.11.2018
%
% See also

% check user defined value for theta, otherwise assign default value
if(nargin < 4)
    theta = 0;
end

% check user defined value for stochasticity, otherwise assign default value
if(nargin < 5)
    stochasticity = 'GaussianNoise';
end

% if user defined value for rngSeed is supplied, set global random seed
if(nargin == 6)
    rng(rngSeed);
end

if(theta > 0)
    
    % modify variables to alter function and gradient evaluation
    
    switch stochasticity
        case 'GaussianNoise'
            % add Gaussian noise to right hand side
            y = y + theta * max(abs(y(:))) * randn(size(y));
        case 'SubSampling'
            [m, n] = size(Gmat);
            rowInd = randperm(m);
            rowInd = rowInd(1:(m - floor((theta - eps(theta)) * m)));
            y      = sqrt(m/length(rowInd)) * y(rowInd);
            Gmat   = sqrt(m/length(rowInd)) * Gmat(rowInd, :);
        otherwise
            notImpErr
    end
    
end

% compute Fx and gradient
res    = Gmat * x - y;
Fx     = 0.5 * sumAll(res.^2);
gradFx = Gmat' * res;

end