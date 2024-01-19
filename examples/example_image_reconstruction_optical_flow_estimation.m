%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script demonstrates the use of the functions for joined image
% reconstruction and motion estimation via optical flow motion models. I  
% described the broad idea in "Enhancing Compressed Sensing Photoacoustic 
% Tomography by Simultaneous Motion Estimation" by Arridge, Beard, Betcke, 
% Cox, Huynh, Lucka Zhang, 2018. but the algorithm here does not split of 
% the forward operator A off. The large scale non-linear optical flow part 
% follows "Joint Large Scale Motion Estmation and Image Reconstruction" by
% Hendrink Dirks, 2018.
% 
% For a given dynamic inverse problem f_t = A_t u_t + epsilon_t, we look at
% solutions that minimize the following energy:
%   (u,v) = argmin { \sum_t^T  1/2 \| A_t u_t - f_t \|_2^2 + \alpha_t \|\nabla u_t \|_1
%                + \beta_t \sum_i^d \|\nabla v_{x_i,t} \|_1
%                + \gamma_t^p/p \| rho(u_{t+1}, u_t, v_t) \|_p^p }
%   OR if symmetric_OF = true
%   (u,v) = argmin { \sum_t^T  1/2 \| A_t u_t - f_t \|_2^2 + \alpha_t \|\nabla u_t \|_1
%                + \beta_t \sum_i^d \|\nabla v_{x_i,t} \|_1
%                + \gamma_t^p/(2*p) \| rho(u_{t+1}, u_t, v_t) \|_p^p
%                + \gamma_t^p/(2*p) \| rho(u_{t}, u_{t+1}, -v_t) \|_p^p}
%
%   where rho(u_{t+1}, u_t, v_t) = u_{t+1} - u_t + (\nabla u_{t+1}) \cdot
%   v_t (OF_type = 'linear', see below)
%      or rho(u_{t+1}, u_t, v_t) = warp(u_{t+1} v_t) - u_t. (OF_type = nonLinear, see below) 
% In addition, non-negativiy constraints u >= 0 will be used. 
%   The algorithm is implemented in TVTVOF_Deblurring.m and alternates 
%   minimization over u and v. Each sub-problem in u and v can
%   be solved via PDGH or ADMM. For a general reference, we refer to
%   "An introduction to continuous optimization for imaging" by Chambolle and
%   Pock, 2016.
%
% In the script, setting certain parameters to 0 will lead to other functions  
% being called:
%   alpha = gamma_p = 0:  will lead to 
%       u_t = argmin_u {1/2 \| A_t u_t - f_t \|_2^2} 
%       being solved by accelerated projected gradient descent for each t
%       (see ProxGradLinearLeastSquares)
%   gamma_p = 0: will lead to 
%       u_t = argmin_u {1/2 \| A_t u_t - f_t \|_2^2} + \alpha_t \|\nabla u_t \|_1
%       being solved by primal dual hybrid gradient for each t (see TV_Deblurring.m)
%   beta = inf will lead to 
%       u = argmin { \sum_t^T  1/2 \| A_t u_t - f_t \|_2^2 + \alpha_t \|\nabla u_t \|_1
%                + \gamma_t^p/p \| u_{t+1} - u_t \|_p^p }
%       being solved by ADMM (see TVOF_Deblurring_ADMM.m, v is set to 0)
%   beta = inf and motion_oracle = true:  will cause the script to compute 
%   a ground truth optical flow solution v_t the ground truth image and solve 
%   u = argmin { \sum_t^T  1/2 \| A_t u_t - f_t \|_2^2 + \alpha_t \|\nabla u_t \|_1
%                + \gamma_t^p/p \| rho(u_{t+1}, u_t, v_t) \|_p^p }
%
% USAGE: Make sure that FelixMatlabTools and FelixMatlabSolvers are on the path!
%        For modality = 'CT', you also need astra and the spot toolbox
%
% ABOUT:
% 	author          - Felix Lucka
% 	date            - 29.10.2023
%  	last update     - 19.01.2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc
rng(1) % reset randon seed for reproducable results

%% set up the scenario

% dynamic imaging modality
modality = 'CT';

% choose between 'twoSpheres', 'squareSpherePhantom', 'squareSphereTriangle'
dynamic_scenario = 'squareSphereTriangle';

% choose image size
n_x     = 100;

%%% choose regularization parameters
% some general hints on how to choose good parameters here:
% 1) first set alpha, beta and gamma_p to 0 and max_iter to 100 to see how
% the unregularized solution looks like 
% 2) set max_iter to 10^3 and choose an alpha > 0 that has a clear visual
% effect on noise and sharpness without smoothing too much
% 3) set max_iter to 10^4, beta to inf and choose a gamma_p that introduces temporal
% smoothing (make sure the triangle gets sharp)
% 4) set motion_oracle = true to see if using a ground truth motion field
% improves the moving parts of the image (it can't get better than this) 
% 5) now set motion_oracle = false, max_iter to 20 and beta = alpha to run
% the joint image reconstruction and motion estimation. Choose higher or
% lower betas depending on whether the resulting optical flow fields look
% too heterogenous or homogenous.
max_iter     = 20;
alpha        = 1*10^(-4); % e.g., denoising: 5*10^(-2) for n_x = 40, CT: 1*10^(-4) for n_x = 100
beta         = 2*10^(-4); % e.g., denoising: 1*10^(-2) for n_x = 40; CT: 2*10^(-4) for n_x = 100
gamma_p      = 1*10^(-2); % e.g., denoising: 1*10^(-1) for n_x = 40; CT: 1*10^(-2) for n_x = 100
p            = 2; % parameters given above are for p = 1
gamma        = (gamma_p).^(1/p);

motion_oracle = false;

OF_type       = 'nonLinear'; % 'linear' or 'nonLinear' (default)
symmetric_OF  = true;
noise_level   = 0.01; % denoising: 0.1; CT: 0.01


%% set up numerical phantom and geometry

n_t    = 5;
expo   = 1;
switch dynamic_scenario
    case 'twoSpheres'
        u_true  = twoSpheres(n_x, 0.3, n_t, expo, []);
    case 'squareSpherePhantom'
        u_true  = squareSpherePhantom(n_x, 0.3, n_t, expo, []);
    case 'squareSphereTriangle'
        u_true  = squareSphereTriangle(n_x, 0.3, n_t, expo, []);
    otherwise
        notImpErr
end

im2vec = @(x) x(:);
vec2im = @(x) reshape(x, [n_x, n_x]);

% visualize phantom
visu_para = [];
visu_para.colorMap       = 'parula';
visu_para.scaling        = 'histCutOffPos';
visu_para.histCutOff     = 1/1000;
visu_para.fps            = 2;
visu_para.loop           = 1;
visu_para.print          = true;
visu_para.fileName       = [dynamic_scenario '_t$frame$'];
%visualizeImage(spaceTime2DynamicData(u_true), [], visu_para)

%% setup dynamic imaging modality


switch modality

    case 'denoising'
        %%% a simple denoising setting, i.e., A_t = Id for all t

        for t = 1:n_t
            A{t}   = @(x) vec(x);
            AT{t}  = @(x) vec2im(x);
            ATA{t} = @(x) x;
        end

    case 'CT'
        %%% a simple 2D parallel beam geo

        % create reconstruction geometry
        vol_geom = astra_create_vol_geom(n_x, n_x);

        % projection geometry
        detector_size = 2  * n_x;
        pixel_size    = 1;

        angular_span_red_fac = 1;
        n_full_angles        = 2*n_x;
        full_angles          = linspace2(0, pi/angular_span_red_fac, n_full_angles);
        angle_partion        = reshape(full_angles, [], n_t);

        for t = 1:n_t
            proj_geom = astra_create_proj_geom('parallel', pixel_size, detector_size, ...
                angle_partion(:, t));

            proj_id   = astra_create_projector('strip', proj_geom, vol_geom);
            matrix_id = astra_mex_projector('matrix', proj_id);
            A_mat{t} = astra_mex_matrix('get', matrix_id);

            ATA_PI = @(x) A_mat{t}' * (A_mat{t} * x);
            lip_constant(t) = powerIteration(ATA_PI, [n_x^2, 1], 10^-10, 1, 0);

            %A{iSS} = opTomo('line_fanflat', projGeom, volGeom);
            astra_mex_projector('delete', proj_id);
            astra_mex_matrix('delete', matrix_id);
        end
        mem(A_mat)

        % normalize by largest lipschitz constant
        normalization_A   = sqrt(max(lip_constant));
        for t = 1:n_t
            A{t}   = @(x) A_mat{t} * (vec(x)/normalization_A);
            AT{t}  = @(x) vec2im(A_mat{t}' * x)/normalization_A;
            ATA{t} = @(x) vec2im(A_mat{t}' * (A_mat{t} * vec(x)))/normalization_A^2;
        end

end
f = dynamicOperator(A, u_true);

% add noise
f = f + noise_level * max(abs(f(:))) * randn(size(f));


%% compute ground truth motion fields between noise free images

if(motion_oracle && ~exist('v_oracle', 'var'))

    % choose beta for this estimation
    beta_here = 3*10^(-2)

    para_ADMM = [];
    para_ADMM.p                = 2;
    para_ADMM.maxEval          = 1000;
    para_ADMM.minEval          = para_ADMM.maxEval;
    para_ADMM.output           = false;
    para_ADMM.rho              = 10^0;
    para_ADMM.rhoAdaptation    = true;
    para_ADMM.rhoAdaptK        = 25;
    para_ADMM.overRelaxPara    = 1.8;
    para_ADMM.lsSolverPara               = [];
    para_ADMM.lsSolverPara.lsSolver      = 'AMG-CG';
    para_ADMM.lsSolverPara.stopCriterion = 'progRelRes';
    para_ADMM.lsSolverPara.progMode      = 'poly';
    para_ADMM.lsSolverPara.tol           = 10^-4;
    para_ADMM.lsSolverPara.tolDecExp     = 1.5;
    para_ADMM.lsSolverPara.minIter       = 3;
    para_ADMM.lsSolverPara.maxIter       = 100;

    pyramid_para = [];
    pyramid_para.algo = para_ADMM;
    pyramid_para.p      = p;
    pyramid_para.symmetricOF = true;
    pyramid_para.output = true;
    pyramid_para.minSz    = 20;
    pyramid_para.maxWarp  = 2;
    pyramid_para.downSamplinggFac = 0.95;
    pyramid_para.computeEnergy    = true;
    pyramid_para.returnBest     = true;
    %pyramidPara.levelIni       = 'last';
    %pyramidPara.lineSearchMode = 'coarse';

    v_oracle = zeros(n_x, n_x, 2, n_t-1);
    for t=1:n_t-1
        [v_oracle(:,:,:,t), infoOFClean{t}] = opticalFlowEstimation(u_true(:,:,t), u_true(:,:,t+1),...
            beta_here, pyramid_para);
    end

    motion_visu_para = [];
    motion_visu_para.scaling      = 'histCutOffPos';
    motion_visu_para.hist_cut_off = 1/1000;
    motion_visu_para.colorVisu    = 'frame';
    motion_visu_para.show         = false;
    for t=1:n_t-1
        u1 = u_true(:,:,t);
        u2 = u_true(:,:,t+1);
        [sum(abs(u1(:) - u2(:)).^2)/2, sum(abs(u1(:) - vec(warpImage(u2, squeeze(v_oracle(:,:,:,t))))).^2)/2]
        [~,~,RGB{t}]    = visualizeMotion(squeeze(v_oracle(:,:,:,t)), ...
            struct('dim', 2, 'type', 'static'), motion_visu_para);
    end

    figure();
    for t=1:n_t-1
        subplot(3,n_t-1,t)  ; imagesc(u_true(:,:,t)); title('u true')
        subplot(3,n_t-1,t+(n_t-1)); imagesc(warpImage(u_true(:,:,t+1),...
            squeeze(v_oracle(:,:,:,t)))); title('u warped')
        subplot(3,n_t-1,t+2*(n_t-1)); image(RGB{t});
    end
    drawnow();

end

%% set up inverse method and numerial optimization scheme

algo_para = [];
algo_para.maxIter         = max_iter;
algo_para.output          = true;
% we use non-negativity constraints here to show how they can be used
algo_para.constraint      = 'nonNegative';
algo_para.displayWarnings = false;

switch OF_type
    case 'linear'
        OF_id = 'l';
    case 'nonLinear'
        if(symmetric_OF)
            OF_id = 'nls';
        else
            OF_id = 'nl';
        end
end

if(alpha > 0)

    if(gamma > 0)

        if(~isinf(beta))

            rec_id     = ['LS-NNTVTVOF'];

            rec_para_id = [OF_id '-a' num2str(alpha,'%.2e') ...
                'b' num2str(beta,'%.2e') 'g' num2str(gamma,'%.2e') 'p' num2str(p)];

            if(motion_oracle)
                error('motion Oracle not possible for this configuration')
            end

            %%% general parameter of the image/motion alternation
            % see TVTVOF_Deblurring.m
            algo_para.maxCompTime     = 48*3600;
            algo_para.hardCompRestr   = true;
            algo_para.output          = true;
            algo_para.monotoneEnergy  = true;
            algo_para.OFtype          = OF_type;
            algo_para.symmetricOF     = symmetric_OF;
            algo_para.p               = p;
            switch p
                case 2
                    algo_para.uIniMode        = 'TV-FbF';
                case 1
                    algo_para.uIniMode        = 'TV-FbF';
            end
            %%% image estimation para
            algo_para.uOpt            = [];
            algo_para.uOpt.solver     = 'ADMM';
            algo_para.uOpt.algo       = [];
            algo_para.uOpt.algo.ATA   = @(x) dynamicOperator(ATA, x);
            algo_para.uOpt.algo.rho           = 10^0;
            algo_para.uOpt.algo.maxEval       = 2*10^3;
            algo_para.uOpt.algo.rhoAdaptation = true;
            algo_para.uOpt.algo.rhoAdaptK     = 25;
            algo_para.uOpt.algo.overRelaxPara = 1.8;
            algo_para.uOpt.algo.output        = false;
            algo_para.uOpt.algo.lsSolverPara  = [];
            algo_para.uOpt.algo.lsSolverPara.lsSolver      = 'CG';
            algo_para.uOpt.algo.lsSolverPara.stopCriterion = 'progRelRes';
            algo_para.uOpt.algo.lsSolverPara.progMode      = 'poly';
            algo_para.uOpt.algo.lsSolverPara.tol           = 10^-3;
            algo_para.uOpt.algo.lsSolverPara.tolDecExp     = 1.5;
            algo_para.uOpt.algo.lsSolverPara.minIter       = 3;
            algo_para.uOpt.algo.lsSolverPara.maxIter       = algo_para.uOpt.algo.maxEval;

            %%% motion estimation para
            algo_para.vOpt                  = [];
            algo_para.vOpt.solver           = 'ADMM';
            algo_para.vOpt.parallel         = true;
            algo_para.vOpt.nWorkerPool      = min(n_t-1,2*maxNumCompThreads());
            % pyramid parameter for non-linear optical flow estimation
            algo_para.vOpt.downSamplinggFac = 0.95;
            algo_para.vOpt.minSz            = 20;
            algo_para.vOpt.maxWarp          = 1;
            algo_para.vOpt.warpTol          = 10^-3;
            algo_para.vOpt.output           = false;
            algo_para.vOpt.algo             = [];
            algo_para.vOpt.algo.output           = false;
            algo_para.vOpt.algo.maxEval          = 500;
            algo_para.vOpt.algo.rho              = 10^0;
            algo_para.vOpt.algo.rhoAdaptation    = false;
            %denoisingAlgo.vOpt.rhoAdaptK  = 25;
            algo_para.vOpt.algo.overRelaxPara    = 1.8;
            algo_para.vOpt.algo.returnAlgoVar    = true;
            algo_para.vOpt.algo.lsSolverPara               = [];
            algo_para.vOpt.algo.lsSolverPara.lsSolver      = 'AMG-CG';
            algo_para.vOpt.algo.lsSolverPara.stopCriterion = 'progRelRes';
            algo_para.vOpt.algo.lsSolverPara.progMode      = 'poly';
            algo_para.vOpt.algo.lsSolverPara.tol           = 10^-4;
            algo_para.vOpt.algo.lsSolverPara.tolDecExp     = 1.5;
            algo_para.vOpt.algo.lsSolverPara.minIter       = 3;
            algo_para.vOpt.algo.lsSolverPara.maxIter       = 100;

            algo_id = ['AA-' int2str(algo_para.maxIter)];

        else

            rec_id     = 'LS-NNTVOF';

            if(motion_oracle)
                rec_para_id = [OF_id '-motionGT-a' num2str(alpha,'%.2e') 'g' num2str(gamma,'%.2e') 'p' num2str(p)];
                algo_para.v = v_oracle;
            else
                rec_para_id = [OF_id '-a' num2str(alpha,'%.2e') 'g' num2str(gamma,'%.2e') 'p' num2str(p)];
            end

            %%% general parameters
            algo_para.maxCompTime     = 1;
            algo_para.hardCompRestr   = true;
            algo_para.maxIter         = 1;
            algo_para.output          = false;
            algo_para.OFtype          = OF_type;
            algo_para.p               = p;
            %%% image estimation para
            algo_para.uOpt               = [];
            algo_para.uOpt.solver     = 'ADMM';
            algo_para.uOpt.algo       = [];
            algo_para.uOpt.algo.ATA   = @(x) dynamicOperator(ATA, x);
            algo_para.uOpt.algo.rho           = 1;
            algo_para.uOpt.algo.maxEval       = max_iter;
            algo_para.uOpt.algo.rhoAdaptation = true;
            algo_para.uOpt.algo.rhoAdaptK     = 25;
            algo_para.uOpt.algo.overRelaxPara = 1.8;
            algo_para.uOpt.algo.output        = true;
            algo_para.uOpt.algo.lsSolverPara  = [];
            algo_para.uOpt.algo.lsSolverPara.lsSolver      = 'CG';
            algo_para.uOpt.algo.lsSolverPara.stopCriterion = 'progRelRes';
            algo_para.uOpt.algo.lsSolverPara.progMode      = 'poly';
            algo_para.uOpt.algo.lsSolverPara.tol           = 10^-3;
            algo_para.uOpt.algo.lsSolverPara.tolDecExp     = 1.5;
            algo_para.uOpt.algo.lsSolverPara.minIter       = 10;
            algo_para.uOpt.algo.lsSolverPara.maxIter       = max_iter/2;

            algo_id = ['A-' num2str(max_iter, '%.g')];

        end

    else

        rec_id      = 'LS-NNTV';
        rec_para_id = ['a' num2str(alpha,'%.2e')];
        algo_id     = ['PD-' int2str(max_iter)];

        switch modality
            case 'denoising'
                algo_para.acceleration         = true;
                algo_para.uniConvexFunctional  = 'F*';
                algo_para.convexityModulus     = 1;
                %algo_para.restart              = false;
                %algo_para.weightFac            = 10;
            case 'CT'
                algo_para.acceleration         = false;
                %algo_para.uniConvexFunctional  = 'F*';
                %algo_para.convexityModulus     = 1;
                %algo_para.restart              = false;
                %algo_para.weightFac            = 10;
        end

    end

else

    if(gamma > 0)

        if(isinf(beta))

            rec_id     = 'LS-NNOF';

            if(motion_oracle)
                rec_para_id = [OF_type '-motionGT-g' num2str(gamma,'%.2e') 'p' num2str(p)];
                algo_para.v = v_oracle;
            else
                rec_para_id = [OF_type '-g' num2str(gamma,'%.2e') 'p' num2str(p)];
            end

            % general parameter of the alternation
            algo_para.maxCompTime     = 1;
            algo_para.hardCompRestr   = true;
            algo_para.maxIter         = 1;
            algo_para.output          = false;
            algo_para.OFtype          = OF_type;
            algo_para.p               = p;
            % u para
            algo_para.uOpt               = [];
            algo_para.uOpt.solver     = 'ADMM';
            algo_para.uOpt.algo       = [];
            algo_para.uOpt.algo.ATA   = @(x) dynamicOperator(ATA, x);
            algo_para.uOpt.algo.rho           = 1;
            algo_para.uOpt.algo.maxEval       = max_iter;
            algo_para.uOpt.algo.rhoAdaptation = true;
            algo_para.uOpt.algo.rhoAdaptK     = 25;
            algo_para.uOpt.algo.overRelaxPara = 1.8;
            algo_para.uOpt.algo.output        = true;
            algo_para.uOpt.algo.lsSolverPara  = [];
            algo_para.uOpt.algo.lsSolverPara.lsSolver      = 'CG';
            algo_para.uOpt.algo.lsSolverPara.stopCriterion = 'progRelRes';
            algo_para.uOpt.algo.lsSolverPara.progMode      = 'poly';
            algo_para.uOpt.algo.lsSolverPara.tol           = 10^-3;
            algo_para.uOpt.algo.lsSolverPara.tolDecExp     = 1.5;
            algo_para.uOpt.algo.lsSolverPara.minIter       = 10;
            algo_para.uOpt.algo.lsSolverPara.maxIter       = max_iter/2;

            algo_id = ['A-' num2str(max_iter, '%.g')];

        else
            notImpErr
        end

    else

        rec_id = 'LS-NN';
        rec_para_id = '';
        algo_id = ['aProxGrad-' int2str(max_iter)];

        % we use projected gradient descend
        algo_para.J            = @(res) 0;
        algo_para.Jx           = 0;
        algo_para.fastGradient = true;
        Proxy = @(x, nu, s, update) projBoxConstraints(x,  'nonNegative');
        nu       = 1; % we normalized A

    end

end

%% compute reconstruction

disp(['computing reconstruction "' rec_id '", with parameter id "' ...
    rec_para_id '" and algo id "' algo_id '"'])

if(gamma == 0)

    % frame by frame reconstruction
    rec      = cell(1, n_t);
    info_opt = cell(1, n_t);
    for t = 1:n_t
        A_t    = A{t};
        Aadj_t = AT{t};
        f_t = f(:, t);
        if(alpha > 0)
            % primal dual hybrid gradient
            [rec{t}, ~, ~, info_opt{t}]  = TV_Deblurring(A_t, Aadj_t, f_t, alpha, algo_para);
        else
            % projected gradient algorithm
            [rec{t}, ~, ~, ~, ~, ~, info_opt{t}] = ProxGradLinearLeastSquares(...
                A_t, Aadj_t, f_t, @(x) x, Proxy, nu, zeros([n_x, n_x]), ...
                zeros(size(f_t)), vec2im(Aadj_t(f_t)), algo_para);
        end
    end

else

    dynA  = @(x) dynamicOperator(A, x);
    dynAT = @(f) dynamicOperator(AT, f);

    [rec, motion, info_opt]  =  TVTVOF_Deblurring(dynA, dynAT, f,...
        alpha, beta, gamma, algo_para);

    rec    = spaceTime2DynamicData(rec);
    motion = spaceTime2DynamicData(motion);

end


%% visulization

print_name = [rec_id '_' rec_para_id '_' algo_id]
visu_para.print          = true;
visu_para.fileName       = [print_name '_t$frame$'];
visu_para.endFrame       = ceil(n_t/2);
visu_para.animatedGif    = isunix();

visualizeImage(rec, [], visu_para)


if(exist('motion','var'))

    motion_visu_para = [];
    motion_visu_para.scaling      = 'histCutOffPos';
    motion_visu_para.hist_cut_off = 1/1000;
    motion_visu_para.colorVisu    = 'frame';
    motion_visu_para.print        = true;
    motion_visu_para.animatedGif  = isunix();
    motion_visu_para.fileName     = [rec_id '_v_t$frame$'];
    motion_visu_para.fps          = 2;
    motion_visu_para.fpsMovie     = 1;
    motion_visu_para.fontSize     = 5;
    motion_visu_para.endFrame     = visu_para.endFrame;
    motion_visu_para.addFrameId   = false;

    visualizeMotion(motion, struct('dim', 2, 'type', 'static'), motion_visu_para)

end

%%

switch modality
    case 'CT'
        astra_mex_data2d('clear');
        astra_mex_data2d('info');
end
