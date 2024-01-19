function [u, info]  = smoothTV_Segmentation(f, alpha, para)
% SMOOTHTVSEGMENTATION implements a multi-class segmentation with a smoothed 
% TV term as a regularizer based on gradient descent type methods
%
% DESCRIBTION:
%   smoothTV_Segmentation performs a multi-class segmentation with a 
%   smoothed TV term based on gradient descent type methods. It solves
%   sum_k^nC < u_k , f_k> + alpha * sum_k^nC H_epsilon(D u_k) such that 
%   sum_nC u_k = 1 everywhere, and u >= 0 
%   In the above,
%   H_\epsilon(D u_k) is given as
%   H_epsilon   = sumAllPixel( sqrt((Dx*u_k).^2 + (Dy*u_k).^2 + epsilon^2) -
%   epsilon)
%   or
%   H_epsilon = sumAllPixel( h_\epsilon(sqrt((Dx*u_k).^2 + (Dy*u_k).^2) ),
%   where h_\epsilon is the Huber function
%   h_\epsilon() = t^2/e*epsilon for t < epsilon and |t|-epsilon/2
%   otherwise
%
% INPUT:
%   f      - data: Intensity based segmentation weights. For a 2D image g of
%            size [n_x, n_y] it's size is given as [n_x, n_y, n_c], where n_c is 
%            number of classes. For instance, if we assume that 
%            we assume that the intensities of each class i follows a 
%            Gaussian with mean c_i and std sigma_i, a good choice of f(i,j,k) would be
%            1/(2*sigma_i^2)(g(i,j) - c_k)^2  
%   alpha  - regularization parameter
%   para   - a struct containing additional parameters:
%     'functional' - 'Huber' (default) or 'SqrtEpsilon'
%     'epsilon' - smoothing parameter
%     'BC' - type of boundary conditions applied
%
% OUTPUTS:
%   u - class probabilities [n_x, n_y, n_c]
%   info  - some information on the iteration
%
% ABOUT:
%   author          - Felix Lucka
%   date            - 25.11.2020
%   last update     - 21.12.2023
%
% See also



% check user defined value for para, otherwise assign default value
sz_u     = size(f);
dim      = length(sz_u)-1;
n_c      = sz_u(end);


functional   = checkSetInput(para, 'functional', {'Huber', 'SqrtEpsilon'}, 'Huber');
epsilon      = checkSetInput(para, 'epsilon', 'double', [], 'error');
BC           = checkSetInput(para, 'BC', {'none','NB','0NB','0','periodic'}, 'NB');


%%% construct parts of the function evaluation and gradient
[D, DT] = FiniteForwardDifferenceOperators(dim+1, BC);

Proj = @(u, nu, proxRes) projProbabilities(u);
nu   = 10^-2;
u0   = getfield(Proj(1./f), 'x');

opt_para    = para;
opt_para.Jx = 0;
opt_para.stepsizeAdaptation = true;

% call ProximalGradientDescent.m to solve the optimization problem
[u, ~, ~, ~, ~, ~, info] = ProximalGradientDescent(@(u) FGrad(u), Proj, nu, u0, opt_para);


    function [Fu, grad_Fu] = FGrad(u)
        
        %%% gradient of the data term is just the weights
        grad_Fu = f;
        Fu      = sum(f(:) .* u(:));
        
        %%%  TV terms
        Du      = cell(dim,1);
        Du_sq    = zeros(size(u), 'like', u);
        for i=1:dim
            Du{i} = D{i}(u);
            Du_sq  = Du_sq + Du{i}.^2;
        end
        grad_Hu = zeros(size(u), 'like', u);
        
        switch functional
            case 'Huber'
                sqrt_Du_sq = sqrt(Du_sq);
                aux      = (sqrt_Du_sq < epsilon);
                Hu       = (Du_sq / (2*epsilon)) .* aux + (sqrt_Du_sq - epsilon/2) .* (~aux);
                for i=1:dim
                    grad_Hu = grad_Hu + DT{i}(Du{i} ./ max(epsilon, sqrt_Du_sq));
                end
            case 'SqrtEpsilon'
                Hu     = sqrt(Du_sq + epsilon^2);
                for i=1:dim
                    grad_Hu = grad_Hu + DT{i}(Du{i} ./ Hu);
                end
        end
        Fu     = Fu     + alpha * sum(Hu(:));
        grad_Fu = grad_Fu + alpha * grad_Hu;
        
    end

end