function [u, y, info]  = TVOF_Deblurring_PDHG(A, AT, f, v, alpha, gamma, para)
%TVOF_DEBLURRING_PDHG uses the primal dual hybrid gradient algorithm to do 
% image deblurring regularized by TV and an optical flow transport term
%
% DESCRIPTION:
%   TVOF_Deblurring_PDHG uses PDHG to do image deblurring regularized by
%   TV and an optical flow transport term almost as described in
%   "Enhancing Compressed Sensing Photoacoustic Tomography by Simultaneous
%   Motion Estimation" by Arridge, Beard, Betcke, Cox, Huynh, Lucka Zhang,
%   2018,
%   but in the above paper, only denoising and a linearize optical flow term
%   was outlined.
%
%   The algorithm tries to minimize the following energy:
%   u = argmin { \sum_t^T  1/2 \| A_t u_t - f_t \|_2^2 + \alpha \|\nabla u_t \|_1
%                + \gamma^p/p \| rho(u_{t+1}, u_{t}, v_t) \|_p^p }
%   OR if symmetricOF = true:
%   u = argmin { \sum_t^T  1/2 \| A_t u_t - f_t \|_2^2 + \alpha_t \|\nabla u_t \|_1
%                + \gamma_t^p/(2*p) \| rho(u_{t+1}, u_t, v_t) \|_p^p
%                + \gamma_t^p/(2*p) \| rho(u_{t}, u_{t+1}, -v_t) \|_p^p}
%   where rho(u_{t+1}, u_t, v_t) = u_{t+1} - u_t + (\nabla u_{t+1}) \cdot v_t
%      or rho(u_{t+1}, u_t, v_t) = warp(u_{t+1} v_t) - u_t
%
%   For a general reference on PDHG, we refer to "An introduction to 
%   continuous optimization for imaging" by Chambolle and Pock, 2016.
%
% USAGE:
%   [u, info] = TVOF_Deblurring_PDHG(A, AT, f, v, alpha, gamma, para)
%
% INPUTS:
%   A      - function handle to the forward operator
%   AT     - function handle to the adjoint of the forward operator
%   f      - the data f as a cell array 
%   v - 2D+1D+1D or 3D+1D+1D motion field
%   alpha - regularization parameter for spatial TV
%   gamma - regularization parameter for transport term
%
%   para  - struct containing additional parameters:
%     'dimU'  - dimension of u as [nX, nX, T] or [nX, nX, nZ, T]
%     'u' - inital value for u
%     'y' - cell with initial values for dual variables
%     'OFtype' - 'linear' or 'nonLinear' depending whether a linearization
%     of the optical flow equation in v is used or not (it is always linear
%     in u)
%     'TVtype' - total variation type used for motion field
%     'p'      - index of the Lp norm, see above
%     'symmetricOF' - boolean determining whether the OF part is
%       symmetric in u_1 and u_2 by adding two OF terms (see above, default: false)
%     'constraint' - constraint on image: 'none' (df), 'positivity' 'range'
%     'dt'      - the time differences between frames as a vector
%       (size T-1 x 1) which will be used to weight the temporal finite
%       differences everywhere.
%     'maxEval' - maximal number of operator evaluations
%     'dataCast' - numerical precision and computing unit of main variables:
%                  'single','double' (df),'gpuArray-single','gpuArray-double'
%
%        'output' - boolean determining whether output should be displayed (df: true)
%        'outputPreStr' - pre-fix for output str (to structure output in multi-layered algos);
%        'outputFreq' - computational time in sec after which new output
%                       is displayed (to limit output in small-scale scenarios)

%
% OUTPUTS:
% 	uBest - image with lowest energy reconstructed 
%   u     - final primal variable u
%   y     - final dual variable y
% 	info  - struct which stores details and variables of the iteration,
%               can be used to inform subsequent calls to this function
%
% ABOUT:
% 	author          - Felix Lucka
% 	date            - 06.10.2020
% 	last update     - 21.12.2023
%
% see also TV_Deblurring_PDHG.m, TVOF_Deblurring_ADMM.m,
% PrimalDualHybridGradient.m


% check user defined value for para, otherwise assign default value
if(nargin < 6)
    para = [];
end


%%% read out parameters
sz_u       = checkSetInput(para, 'sizeU', 'i,>0', [], 'error');
dim_space  = length(sz_u) - 1;
n_t        = sz_u(end);

output       = checkSetInput(para, 'output', 'logical', false);
data_cast    = checkSetInput(para, 'dataCast', ...
    {'single','double','gpuArray-single','gpuArray-double','input'}, 'input');
OF_type      = checkSetInput(para, 'OFtype', {'linear', 'nonLinear'}, 'linear');
TV_type      = checkSetInput(para, 'TVtype', {'isotropic'}, 'isotropic');
p            = checkSetInput(para, 'p', '>0', 2);
symmetric_OF = checkSetInput(para, 'symmetricOF', 'logical', false);
constraint   = checkSetInput(para, 'constraint', {'none','nonNegative', 'range'},'none');
weight_fac   = checkSetInput(para, 'weightFac', '>0', 1);
OF_para.dt    = checkSetInput(para, 'dt', '>0', ones(1, n_t-1));
OF_para.warpPara = checkSetInput(para, 'warpPara', 'struct', emptyStruct);
OF_para.dimSpace = dim_space;


%%% construct K
if(~symmetric_OF)
    K.nComponents = 3;
    K.fwd = @(x) {A(x), ...
        spatialFwdGrad(x, false), ...
        opticalFlowOperator(x, v, OF_type, false, OF_para, false)};
    K.adj = @(y)    AT(y{1}) ...
        + spatialFwdGrad(y{2}, true) ...
        + opticalFlowOperator(y{3}, v, OF_type, true, OF_para, false);
else
    K.nComponents = 4;
    K.fwd = @(x) {A(x), ...
        spatialFwdGrad(x, false), ...
        opticalFlowOperator(x, v, OF_type, false, OF_para, false),...
        opticalFlowOperator(x, v, OF_type, false, OF_para, true)};
    K.adj = @(y)    AT(y{1}) ...
        + spatialFwdGrad(y{2}, true) ...
        + opticalFlowOperator(y{3}, v, OF_type, true, OF_para, false) ...
        + opticalFlowOperator(y{4}, v, OF_type, true, OF_para, true);
end


%%% define G and F and their proxies

% F gathers L2 data term, TV regularization on the images and optical flow term
if(~symmetric_OF)
    para.F = @(Kx) gather(1/2 *    sum((Kx{1}(:)-f(:)).^2) ...
        + alpha      * sumAll(sqrt(sum(Kx{2}.^2, dim_space+1))) ...
        + gamma.^p/p * sum(abs(Kx{3}(:)).^p));
    proxF = @(y, lambda) mergeProxRes(proxL2(y{1},  lambda, f), ...
        proxL21(y{2}, lambda * alpha, dim_space+1), ...
        proxLp(y{3},  lambda * gamma^p, p));
else
    para.F = @(Kx) gather(1/2 *    sum((Kx{1}(:)-f(:)).^2) ...
        + alpha   * sumAll(sqrt(sum(Kx{2}.^2, dim_space+1))) ...
        + gamma^p/(2*p) * sum(abs(Kx{3}(:)).^p) ...
        + gamma^p/(2*p) * sum(abs(Kx{4}(:)).^p));
    proxF = @(y, lambda) mergeProxRes(proxL2(y{1},  lambda, f), ...
        proxL21(y{2}, lambda * alpha, dim_space+1), ...
        proxLp(y{3},  lambda * gamma^p/2, p), ...
        proxLp(y{4},  lambda * gamma^p/2, p));
end
                
% G just deals with the constraints on u
switch constraint
    case {'none', 'nonNegative'}
        conRange = [];
    case 'range'
        conRange = checkSetInput(para, 'conRange', 'double', [], 'error');
        if(~ length(conRange) == 2 && conRange(1) < conRange(2))
            error(['invalid range: ' num2str(conRange)])
        end
end
para.G = @(x) energyBoxConstraints(x, constraint, conRange);
proxG  = @(z, lambda) projBoxConstraints(z, constraint, conRange);


%%% sigma, tau and theta of PDHG scheme (automatically determined by default)
sigma = checkSetInput(para, 'sigma', '>0', 'auto');
tau   = checkSetInput(para, 'tau',   '>0', 'auto');
theta = checkSetInput(para,'theta',  '>=0', 1);


%%% cast function
switch data_cast
    case 'input'
        castFun   = @(f) f;
        castZeros = @(sz) zeros(sz, 'like', f);
    otherwise
        [castFun, castZeros] = castFunction(data_cast);
end
f = castFun(f);



%%% initial values for x and y
[u, ~, para] = checkSetInput(para, 'u', 'numeric', castZeros(sz_u), [], true);
u = castFun(u);
Kx = K.fwd(u);

y = castFun(checkSetInput(para, 'y', 'cell', Kx));


%%% some more parameters
para.proxFnotConj        = true; % we use F, not F*
para.maxIter             = checkSetInput(para, 'maxIter', 'i,>0', 100);


%%% run the optimization
[u, y, iter, ~, ~, info]  = PrimalDualHybridGradient(K, proxG, proxF,...
    sigma, tau, theta, u, Kx, y, para);


%%% gather results
info.Jx = gather(para.F(K.fwd(u))/alpha);
u       = gather(u);
y       = gather(y);
info.iterReturn = iter;

end