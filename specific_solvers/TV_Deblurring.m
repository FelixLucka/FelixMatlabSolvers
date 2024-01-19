function [x, y, iter, info]  = TV_Deblurring(A, AT, f, alpha, para)
% TVDEBLURRING implements a TV deblurring based on the primal
% dual hybrid gradient algorithm
%
% DESCRIBTION:
%   TVdenoising performs TV densoing based on the primal dual hybrid gradient
%   algorithm. It solves
%   min_x 1/2 \| W * (A x - f) \|_2^2 + alpha TV(x)
%   and allows for the inclusion of box constraints on x
%
%
% INPUT:
%   A      - function handle to the forward operator
%   AT     - function handle to the adjoint of the forward operator
%   f      - image of dimension 2 or 3 to be denoised
%   alpha  - regularization parameter
%   para   - a struct containing additional parameters:
%     'weighting' - a diagonal weighting of the data term:
%          1/2 \| W * (A x -f) \|_2^2
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
%   date            - 06.05.2018
%   last update     - 27.10.2023
%
% See also


info = [];

% check user defined value for para, otherwise assign default value
if(nargin < 5)
    para = [];
end


%%% read out parameters
[sz_x, cmp_sz_x] = checkSetInput(para, 'sizeX', 'i,>0', 1);
if(cmp_sz_x)
    sz_x = size(AT(f));
end
dim = length(sz_x);

weighting    = checkSetInput(para, 'weighting', 'double', 1);
BC           = checkSetInput(para, 'BC', {'none','NB','0NB','0','periodic'}, 'NB');
coordinates  = checkSetInput(para, 'coordinates', 'i,>0', 1:dim);
constraint   = checkSetInput(para, 'constraint', {'none','nonNegative','range'}, 'none');
output       = checkSetInput(para, 'output', 'logical', false);
data_cast    = checkSetInput(para, 'dataCast', ...
    {'single','double','gpuArray-single','gpuArray-double','input'}, 'input');


%%% get finite forward difference operators and construct K
K.nComponents = 2;
[D, DT] = FiniteForwardDifferenceOperators(dim, BC);
switch length(coordinates)
    case 1
        K.fwd = @(x) {A(x), D{coordinates(1)}(x)};
        K.adj = @(y) AT(y{1}) + DT{coordinates(1)}(y{2});
    case 2
        K.fwd = @(x) {A(x), cat(dim+1, D{coordinates(1)}(x), D{coordinates(2)}(x))};
        K.adj = @(y) AT(y{1}) + DT{coordinates(1)}(sliceArray(y{2}, dim+1, 1, 1)) ...
            + DT{coordinates(2)}(sliceArray(y{2}, dim+1, 2, 1));
    case 3
        K.fwd = @(x) {A(x), cat(dim+1, D{coordinates(1)}(x), D{coordinates(2)}(x), D{coordinates(3)}(x))};
        K.adj = @(y) AT(y{1}) + DT{coordinates(1)}(sliceArray(y{2}, dim+1, 1, 1)) ...
            + DT{coordinates(2)}(sliceArray(y{2}, dim+1, 2, 1)) ...
            + DT{coordinates(3)}(sliceArray(y{2}, dim+1, 3, 1));
    case 4
        K.fwd = @(x) {A(x), cat(dim+1, D{coordinates(1)}(x), D{coordinates(2)}(x), ...
            D{coordinates(3)}(x), D{coordinates(4)}(x))};
        K.adj = @(y) AT(y{1}) + DT{coordinates(1)}(sliceArray(y{2}, dim+1, 1, 1)) ...
            + DT{coordinates(2)}(sliceArray(y{2}, dim+1, 2, 1)) ...
            + DT{coordinates(3)}(sliceArray(y{2}, dim+1, 3, 1)) ...
            + DT{coordinates(4)}(sliceArray(y{2}, dim+1, 4, 1));
    otherwise
        notImpErr
end


%%% choose step sizes
[lipschitz_constant, cmp_lip] = checkSetInput(para, 'LipschitzConstant', '>0', 1);
if(cmp_lip)
    lipP_power_iter_tol  = checkSetInput(para, 'LipPowerIterTol', '>0', 10^-3);
    KTK                  = @(x) K.adj(K.fwd(x));
    
    clock_lip = tic; % track computation time
    [lipschitz_constant, info_lip] = powerIteration(KTK, sz_x, lipP_power_iter_tol, 1, output);
    info.tCompLip = toc(clock_lip); 
    info.LipschitzConstant      = lipschitz_constant;
    info.LipInfo                = info_lip;
    
    return_only_lipschitz = checkSetInput(para, 'returnOnlyLipschitz', 'logical', false);
    if(return_only_lipschitz)
        x = []; y = []; iter = 0;
        return
    end
end

weight_fac = checkSetInput(para, 'weightFac', '>0', 1);
% sigma and tau of PDHG scheme
sigma = checkSetInput(para, 'sigma', '>0', weight_fac*sqrt(0.9/lipschitz_constant));
tau   = checkSetInput(para, 'tau', '>0',   sqrt(0.9/lipschitz_constant)/weight_fac);
if(lipschitz_constant * sigma * tau > 1)
    warning('choice of sigma and tau might lead to too big step sizes!')
end
theta = checkSetInput(para,'theta','>=0',1);


%%% cast function
switch data_cast
    case 'input'
        castFun   = @(f) f;
    otherwise
        castFun = castFunction(data_cast);
end
f         = castFun(f);
weighting = castFun(weighting);


%%% define G and F and their proxies
% F gathers L2 data term and regularization, G only constraints
para.F = @(Kx) gather(1/2 * sum((weighting(:) .* (Kx{1}(:)-f(:))).^2) + ...
    alpha * sum(vec(sqrt(sum(Kx{2}.^2, dim+1)))));
prox_F = @(y, lambda) mergeProxRes(proxL2(y{1}, lambda, f, weighting), proxL21(y{2}, lambda * alpha, dim+1));


%%% define proxies
switch constraint
    case {'none', 'nonNegative'}
        con_range = [];
    case 'range'
        con_range = checkSetInput(para, 'conRange', 'double', [], 'error');
        if(~ length(con_range) == 2 && con_range(1) < con_range(2))
            error(['invalid range: ' num2str(con_range)])
        end
end
para.G = @(x) energyBoxConstraints(x, constraint, con_range);
prox_G  = @(z, lambda) projBoxConstraints(z, constraint, con_range);


%%% initial values for x and y
x       = castFun(checkSetInput(para, 'x', 'double', zeros(sz_x)));
Kx      = K.fwd(x);
y       = castFun(checkSetInput(para, 'y', 'cell', Kx));


%%% some more parameters
para.proxFnotConj        = true;
para.maxIter             = checkSetInput(para, 'maxIter', 'i,>0', 100);


%%% run the optimization
[x, y, iter, ~, ~, infoPDHG]  = PrimalDualHybridGradient(K, prox_G, prox_F,...
    sigma, tau, theta, x, Kx, y, para);

info = overwriteFields(info, infoPDHG, true);

%%% gather results
info.Jx = gather(para.F(K.fwd(x))/alpha);
x       = gather(x);
y       = gather(y);
info.iterReturn = iter;
if(cmp_lip)
    info.LipschitzConstant      = lipschitz_constant;
    info.LipInfo                = info_lip;
end

end