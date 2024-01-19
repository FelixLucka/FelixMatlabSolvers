function [u, info]  = smoothTV_Deblurring(A, AT, f, alpha, para)
% SMOOTHTVDEBLURRING implements a deblurring with a smoothed TV term based
% on gradient descent type methods
%
% DESCRIBTION:
%   smoothTV_Deblurring performs deburring with a smoothed TV term based
% on gradient descent type methods. It solves
%   1/2 \| W * (A u - f) \|_2^2 + alpha * H_epsilon(Du)
%   and allows for the inclusion of box constraints. In the above,
%   H_\epsilon is given as
%   H_epsilon   = sumAllPixel( sqrt((Dx*u).^2 + (Dy*u).^2 + epsilon^2) -
%   epsilon)
%   or
%   H_epsilon = sumAllPixel( h_\epsilon(sqrt((Dx*u).^2 + (Dy*u).^2) ),
%   where h_\epsilon is the Huber function
%   h_\epsilon() = t^2/e*epsilon for t < epsilon and |t|-epsilon/2
%   otherwise
%   and allows for the inclusion of box constraints
%
% INPUT:
%   A      - function handle to the forward operator
%   AT     - function handle to the adjoint of the forward operator
%   f      - image of dimension 2 or 3 to be denoised
%   alpha  - regularization parameter
%   para   - a struct containing additional parameters:
%     'functional' - 'Huber' (default) or 'SqrtEpsilon'
%     'epsilon' - smoothing parameter
%     'weighting' - a diagonal weighting of the data term:
%          1/2 \| W * (x -f) \|_2^2
%     by default, it is 1
%     'BC' - type of boundary conditions applied
%     'coordinates' - coordinates along which gradients should be computed
%     (to only smooth along certain dimensions)
%     'constraint' - constraints on x to apply
%     'dataCast' - in which numerical type the computation should be
%     performed (default: in the same form the image is given)
%
% OUTPUTS:
%   u - denoised image
%   info  - some information on the iteration
%
% ABOUT:
%   author          - Felix Lucka
%   date            - 13.11.2020
%   last update     - 21.12.2023
%
% See also



% check user defined value for para, otherwise assign default value
if(nargin < 5)
    para = [];
end


%%% read out parameters
[sz_u, cmp_size_x] = checkSetInput(para, 'szU', 'i,>0', 1);
if(cmp_size_x)
    sz_u = size(AT(f));
end
dim = length(sz_u);

functional   = checkSetInput(para, 'functional', {'Huber', 'SqrtEpsilon'}, 'Huber');
epsilon      = checkSetInput(para, 'epsilon', 'double', [], 'error');
weighting    = checkSetInput(para, 'weighting', 'double', 1);
BC           = checkSetInput(para, 'BC', {'none','NB','0NB','0','periodic'}, 'NB');
data_cast     = checkSetInput(para, 'dataCast', ...
    {'single','double','gpuArray-single','gpuArray-double'}, class(f));
[~,  castZeros]  = castFunction(data_cast, false, true);


%%% construct parts of the function evaluation and gradient
[D, DT] = FiniteForwardDifferenceOperators(dim, BC);

% solve the optimization problem using LBFGS
lbfgs_para = para;
[u, ~, ~, ~, info] = LBFGS(@(u) FGrad(u), castZeros(sz_u), lbfgs_para);

    function [Fu, grad_Fu] = FGrad(u)
        
        %%% data term
        Au      = A(u);
        Fu      = 1/2 * sum((weighting(:) .* (Au(:)-f(:))).^2);
        grad_Fu = AT(weighting.^2 .*(Au - f));
        
        %%%  TV term
        Du       = cell(dim,1);
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