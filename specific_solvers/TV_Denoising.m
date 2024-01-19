function [x, y, iter, info]  = TV_Denoising(f, alpha, para)
% TVDENOISING implements a TV denoising based on the primal 
% dual hybrid gradient algorithm
% 
% DESCRIBTION:
%   TVdenoising performs TV densoing based on the primal dual hybrid gradient 
%   algorithm. It solves
%   min_x 1/2 \| W * (x -f) \|_2^2 + alpha TV(x) 
%   and allows for the inclusion of box constraints on x
%   
%
% INPUT:
%   f      - image of dimension 2 or 3 to be denoised
%   lambda - regularization parameter
%   para   - a struct containing additional parameters:
%     'weighting' - a diagonal weighting of the data term:
%          1/2 \| W * (x -f) \|_2^2
%     by default, it is 1
%     'BC' - type of boundary conditions applied
%     'coordinates' - coordinates along which gradients should be computed
%     (to only smooth along certain dimensions)
%     'constraint' - constraints on x to apply
%     'dataCast' - in which numerical type the computation should be
%     performed (default: in the same form the image is given)
%     'sigma', 'tau', 'theta' - see PrimalDualHybridGradient.m
%     all other parameters are looped through to
%     PrimalDualHybridGradient.m, see the corresponding documentation
%
% OUTPUTS:
%   x - denoised image
%   y - dual variable
%   iter  - number of iterations
%   info  - some information on the iteration
%
% ABOUT:
%   author          - Felix Lucka
%   date            - 16.03.2018
%   last update     - 27.10.2023
%
% See also 


% check user defined value for para, otherwise assign default value
if(nargin < 3)
    para = [];   
end

dim = ndims(f);


%%% read out parameters
weighting    = checkSetInput(para, 'weighting', 'double', 1);
BC           = checkSetInput(para, 'BC', {'none','NB','0NB','0','periodic'}, 'NB');
coordinates  = checkSetInput(para, 'coordinates', 'i,>0', 1:dim);
constraint   = checkSetInput(para, 'constraint', {'none','nonNegative','range'}, 'none');
data_cast    = checkSetInput(para, 'dataCast', ...
    {'single','double','gpuArray-single','gpuArray-double','input'}, 'input');


%%% get finite forward difference operators and construct K
K.nComponents = 1;
[D, DT] = FiniteForwardDifferenceOperators(dim, BC);
switch length(coordinates)
    case 1
        K.fwd = @(x) {D{coordinates(1)}(x)};
        K.adj = @(y) DT{coordinates(1)}(y{1});
    case 2
        K.fwd = @(x) {cat(dim+1, D{coordinates(1)}(x), D{coordinates(2)}(x))};
        K.adj = @(y) DT{coordinates(1)}(sliceArray(y{1}, dim+1, 1, 1)) + DT{coordinates(2)}(sliceArray(y{1}, dim+1, 2, 1));
    case 3
        K.fwd = @(x) {cat(dim+1, D{coordinates(1)}(x), D{coordinates(2)}(x), D{coordinates(3)}(x))};
        K.adj = @(y) DT{coordinates(1)}(sliceArray(y{1}, dim+1, 1, 1)) + ...
            DT{coordinates(2)}(sliceArray(y{1}, dim+1, 2, 1)) + DT{coordinates(3)}(sliceArray(y{1}, dim+1, 3, 1)); 
    case 4
        K.fwd = @(x) {cat(dim+1, D{coordinates(1)}(x), D{coordinates(2)}(x), ...
            D{coordinates(3)}(x), D{coordinates(4)}(x))};
        K.adj = @(y) DT{coordinates(1)}(sliceArray(y{1}, dim+1, 1, 1)) + DT{coordinates(2)}(sliceArray(y{1}, dim+1, 2, 1)) ...
                   + DT{coordinates(3)}(sliceArray(y{1}, dim+1, 3, 1)) + DT{coordinates(4)}(sliceArray(y{1}, dim+1, 4, 1));
    otherwise
        notImpErr
end


%%% choose step sizes
lipschitz_constant = 4*length(coordinates);
weight_fac         = alpha;
% sigma and tau of PDHG scheme
sigma = checkSetInput(para, 'sigma', '>0', weight_fac*sqrt(0.99/lipschitz_constant));
tau   = checkSetInput(para, 'tau', '>0', sqrt(0.99/lipschitz_constant)/weight_fac);
if(lipschitz_constant * sigma * tau > 1)
    warning('choice of sigma and tau might lead to too big step sizes!')
end
theta = checkSetInput(para,'theta','>0',1);

%%% cast function 
switch data_cast
    case 'input'
        cast_fun = @(f) f;
    otherwise
        cast_fun = castFunction(data_cast);
end
f         = cast_fun(f);
weighting = cast_fun(weighting);
 

%%% define G and F
para.F = @(Kx) gather(alpha * sum(vec(sqrt(sum(Kx{1}.^2, dim+1)))));

%%% define proxies
switch constraint
    case 'none'
        para.G = @(x) gather(1/2 * sum((weighting(:) .* (x(:)-f(:))).^2));
        prox_G  = @(z, lambda) proxL2(z, lambda, f, weighting);
    case 'nonNegative'
        para.G = @(x) gather(1/2 * sum((weighting(:) .* (x(:)-f(:))).^2) + (1./not(any(x(:) < 0))-1));
        prox_G = @(z, lambda) proxL2(z, lambda, f, weighting, 0);
    case 'range'
        conRange = checkSetInput(para, 'conRange', 'double', [], 'error');
        if(~ length(conRange) == 2 && conRange(1) < conRange(2))
            error(['invalid range: ' num2str(conRange)])
        end
        para.G = @(x) gather(1/2 * sum((weighting(:) .* (x(:)-f(:))).^2) + (1./not(any(x(:) < conRange(1)))-1)  + (1./not(any(x(:) > conRange(2)))-1));
        prox_G = @(z, lambda) proxL2(z, lambda, f, weighting, conRange(1), conRange(2));
end
prox_F = @(y, lambda) mergeProxRes(proxL21(y{1}, lambda * alpha, dim+1));

%%% initial values for x and y
prox_res = prox_G(f, 0);
x        = cast_fun(checkSetInput(para, 'x', 'double', prox_res.x));
Kx       = K.fwd(x);
y        = cast_fun(checkSetInput(para, 'y', 'cell', {zeros(size(Kx{1}))}));


%%% some more parameters
para.proxFnotConj        = true;
para.maxIter             = checkSetInput(para, 'maxIter', 'i,>0', 100);
para.acceleration        = checkSetInput(para, 'acceleration', 'logical', true);
para.uniConvexFunctional = 'G';
para.convexityModulus    = min(weighting(:).^2);

%%% run the optimization
[x, y, iter, ~, ~, info]  = PrimalDualHybridGradient(K, prox_G, prox_F,...
    sigma, tau, theta, x, Kx, y, para);


info.Jx = gather(para.F(K.fwd(x))/alpha);
x       = gather(x);
y       = gather(y);

end