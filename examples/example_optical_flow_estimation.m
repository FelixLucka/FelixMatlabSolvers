%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script demonstrates the use of the opticalFlowEstimation.m function
%
% USAGE: Make sure that FelixMatlabTools and FelixMatlabSolvers are on the
% path!
%
% ABOUT:
% 	author          - Felix Lucka
% 	date            - 29.10.2023
%  	last update     - 21.12.2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ccc

%% set up the scenario 

% choose image size
n_x     = 50;

% choose between 'small', 'medium', 'large'
motion_scale = 'large';

% relative level of Gaussian noise that will be added 
noise_level   = 0.0;

% regulariation parameter of the optical flow tern
beta         = 1 * 10^-3; % regularization of the optical flow

%% set up visulization parameters 

visu_para = [];
visu_para.colorMap       = 'cool2hot';
visu_para.nonNeg         = false;
visu_para.histCutOff     = 1/1000;
visu_para.docked         = true;

motion_visu_para = [];
motion_visu_para.histCutOff = 1/1000;
motion_visu_para.colorVisu    = 'frame';
motion_visu_para.print        = true;
motion_visu_para.animatedGif  = true;
motion_visu_para.docked       = true;

%% set up numerical phantom and geometry


switch motion_scale
    case 'small'
        n_t      = 100;  % larger means smaller motion
    case 'medium'
        n_t      = 6;  % larger means smaller motion
    case 'large'
        n_t      = 2;  % larger means smaller motion
end

u_true  = squareSphereTriangle(n_x, 0.3, 2*n_t, 1, []);
u1      = u_true(:,:,n_t);
u2      = u_true(:,:,n_t+1);
clear u_true

v_true                        = zeros(n_x, n_x, 2);
v_true(1:ceil(n_x/3),:,2)     = 33/200 * n_x;
v_true(ceil(n_x/3)+1:end,:,1) = 33/200 * n_x;
v_true(ceil(n_x/3)+1:end,:,2) = 146/800 * n_x;
for i=ceil(6*n_x/12):n_x
    for j=1:(i-ceil(6*n_x/12))
        v_true(i,j,:) = 0;
    end
end
 
% add noise
u1 = u1 + noise_level * max(u1(:)) * randn(size(u1));
u2 = u2 + noise_level * max(u2(:)) * randn(size(u2));

figure();
subplot(1,3,1); imagesc(u1);
subplot(1,3,2); imagesc(u2);
subplot(1,3,3); imagesc(u2-u1);
if(exist('v_true', 'var'))
    visualizeMotion(v_true, struct('dim', 2, 'type', 'static'), motion_visu_para);
end

%% set up parameters of ADMM algorithm to solve the optical flow problems 
% (see TV_FlowEstimation_ADMM.m)

para_ADMM = [];
para_ADMM.p                = 2;
para_ADMM.maxEval          = 1000;
para_ADMM.minEval          = para_ADMM.maxEval;
para_ADMM.output           = true;
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

%% my pyramid version
   
pyramid_para = [];
pyramid_para.output = true;
pyramid_para.nLevel = 1;
pyramid_para.maxWarp  = 1;
pyramid_para.downSamplinggFac = 1;
pyramid_para.algo           = para_ADMM;
pyramid_para.computeEnergy  = true;
pyramid_para.returnBest     = true;
pyramid_para.levelIni       = 'last';
pyramid_para.lineSearchMode = 'coarse';

% just solve linerized optical flow equation 
[v_lin, info_lin] = opticalFlowEstimation(u1, u2, beta, pyramid_para);

% solve linerized symmetrical optical flow equation 
pyramid_para.symmetricOF = true;
[v_lin_sym, info_lin_sym] = opticalFlowEstimation(u1, u2, beta, pyramid_para);

% include 3 x warping on same resolution
pyramid_para.symmetricOF = false;
pyramid_para.maxWarp     = 3;
pyramid_para.algo.output = false;
[v_lin_3warp, info_lin_3warp] = opticalFlowEstimation(u1, u2, beta, pyramid_para);

% fast pyramid scheme to solve full nonlinear optical flow equation
pyramid_para                  = removeFields(pyramid_para, {'nLevel', 'symmetricOF'});
pyramid_para.algo.output      = false;
pyramid_para.warpTol          = 10^-3;
pyramid_para.maxWarp          = 3;
pyramid_para.minSz            = 10;
pyramid_para.downSamplinggFac = 0.5;

[v_nonlin_fast, info_nonlin_fast] = opticalFlowEstimation(u1, u2, beta, pyramid_para);
info_nonlin_fast.energy

% slower pyramid scheme to solve full nonlinear optical flow equation
pyramid_para.downSamplinggFac = 0.8;

[v_nonlin, info_nonlin] = opticalFlowEstimation(u1, u2, beta, pyramid_para);
info_nonlin.energy

% slower pyramid scheme to solve symmetical full nonlinear optical flow equation
pyramid_para.symmetricOF = true;
[v_nonlin_sym, info_nonlin_sym] = opticalFlowEstimation(u1, u2, beta, pyramid_para);
info_nonlin_sym.energy

%% evaluation
%clc
close all

% measure l2 error of optical flow equation
norm(u1(:) - u2(:))
norm(u1(:) - vec(warpImage(u2, v_lin)))
norm(u1(:) - vec(warpImage(u2, v_lin_3warp)))
norm(u1(:) - vec(warpImage(u2, v_lin_sym)))
norm(u1(:) - vec(warpImage(u2, v_nonlin_fast)))
norm(u1(:) - vec(warpImage(u2, v_nonlin)))
norm(u1(:) - vec(warpImage(u2, v_nonlin_sym)))

% print motion fields 
info_dummy      = [];
info_dummy.dim  = 2;
info_dummy.type = 'static';
motion_visu_para.show = false;
[~,~,RGB_Lin]        = visualizeMotion(v_lin,    info_dummy, motion_visu_para);
[~,~,RGB_Lin3]       = visualizeMotion(v_lin_3warp,    info_dummy, motion_visu_para);
[~,~,RGB_LinSym]     = visualizeMotion(v_lin_sym, info_dummy, motion_visu_para);
[~,~,RGB_NoLinFast]  = visualizeMotion(v_nonlin_fast,    info_dummy, motion_visu_para);
[~,~,RGB_NoLin]      = visualizeMotion(v_nonlin,    info_dummy, motion_visu_para);
[~,~,RGB_NoLinSym]   = visualizeMotion(v_nonlin_sym, info_dummy, motion_visu_para);


% comparison plts
figure();
subplot(5,7,1);  imagesc(u1); title('u1')
subplot(5,7,8);  imagesc(u2); title('u2')

subplot(5,7,2);  imagesc(warpImage(u2,  v_lin));      title('Warp(u2,  vLin)')
subplot(5,7,3);  imagesc(warpImage(u2,  v_lin_3warp));      title('Warp(u2,  vLin3)')
subplot(5,7,4);  imagesc(warpImage(u2,  v_lin_sym));   title('Warp(u2,  vLinSym)')
subplot(5,7,5);  imagesc(warpImage(u2,  v_nonlin_fast));    title('Warp(u2,  vNoLinFast)')
subplot(5,7,6);  imagesc(warpImage(u2,  v_nonlin));    title('Warp(u2,  vNoLin)')
subplot(5,7,7);  imagesc(warpImage(u2,  v_nonlin_sym)); title('Warp(u2,  vNoLinSym)')

subplot(5,7,9);  imagesc(warpImage(u1, -v_lin));      title('Warp(u1, -vLin)')
subplot(5,7,10);  imagesc(warpImage(u1, -v_lin_3warp));      title('Warp(u1, -vLin3)')
subplot(5,7,11);  imagesc(warpImage(u1, -v_lin_sym));   title('Warp(u1, -vLinSym)')
subplot(5,7,12); imagesc(warpImage(u1, -v_nonlin_fast));    title('Warp(u1, -vNoLinFast)')
subplot(5,7,13); imagesc(warpImage(u1, -v_nonlin));    title('Warp(u1, -vNoLin)')
subplot(5,7,14); imagesc(warpImage(u1, -v_nonlin_sym)); title('Warp(u1, -vNoLinSym)')

subplot(5,7,16); image(RGB_Lin);                     title('vLin')
subplot(5,7,17); image(RGB_Lin3);                  title('vLin3')
subplot(5,7,18); image(RGB_LinSym);                  title('vLinSym')
subplot(5,7,19); image(RGB_NoLinFast);               title('vNoLinFast')
subplot(5,7,20); image(RGB_NoLin);                   title('vNoLin')
subplot(5,7,21); image(RGB_NoLinSym);                title('vLinSym')

subplot(5,7,22);  imagesc(u1 - u2, [-1,1]); title('u1-u2')
subplot(5,7,23);  imagesc(u1 - warpImage(u2, v_lin), [-1,1]);           title('u1- Warp(u2, vLin) ')
subplot(5,7,24);  imagesc(u1 - warpImage(u2, v_lin_3warp) , [-1,1]);     title('u1- Warp(u2, vLin3)')
subplot(5,7,25);  imagesc(u1 - warpImage(u2, v_lin_sym) , [-1,1]);     title('u1- Warp(u2, vLinSym)')
subplot(5,7,26);  imagesc(u1 - warpImage(u2, v_nonlin_fast) , [-1,1]);       title('u1- Warp(u2, vNoLinFast) ')
subplot(5,7,27);  imagesc(u1 - warpImage(u2, v_nonlin) , [-1,1]);       title('u1- Warp(u2, vNoLin) ')
subplot(5,7,28);  imagesc(u1 - warpImage(u2, v_nonlin_sym) , [-1,1]); title('u1- Warp(u2, vNoLinSym) ')

subplot(5,7,30);  imagesc(u1 - warpImage(u2, v_lin) + u2 - warpImage(u1, -v_lin), [-1,1]);           title('u1- Warp(u2, vLin) + u2 - Warp(u1, -vLin)')
subplot(5,7,31);  imagesc(u1 - warpImage(u2, v_lin_3warp) + u2 - warpImage(u1, -v_lin_3warp), [-1,1]);     title('u1- Warp(u2, vLin3) + u2 - Warp(u1, -vLin3)')
subplot(5,7,32);  imagesc(u1 - warpImage(u2, v_lin_sym) + u2 - warpImage(u1, -v_lin_sym), [-1,1]);     title('u1- Warp(u2, vLinSym) + u2 - Warp(u1, -vLinSym)')
subplot(5,7,33);  imagesc(u1 - warpImage(u2, v_nonlin_fast) + u2 - warpImage(u1, -v_nonlin_fast), [-1,1]);       title('u1- Warp(u2, vNoLinFast) + u2 - Warp(u1, -vNoLinFast)')
subplot(5,7,34);  imagesc(u1 - warpImage(u2, v_nonlin) + u2 - warpImage(u1, -v_nonlin), [-1,1]);       title('u1- Warp(u2, vNoLin) + u2 - Warp(u1, -vNoLin)')
subplot(5,7,35);  imagesc(u1 - warpImage(u2, v_nonlin_sym) + u2 - warpImage(u1, -v_nonlin_sym), [-1,1]); title('u1- Warp(u2, vNoLinSym) + u2 - Warp(u1, -vLinSym)')

%% print results

folder = 'OpticalFlowIllustration';
makeDir(folder);

visu_para = [];
visu_para.colorMap       = 'parula';
visu_para.clim           = [0,1];
visu_para.print          = true;
visu_para.show           = false;

visu_para_diff = [];
visu_para_diff.colorMap       = 'blue2red';
visu_para_diff.clim               = [-1,1];
visu_para_diff.print          = true;
visu_para_diff.show           = false;

file_name_pre = [folder '/OpticalFlowIllustration_' motion_scale];

visu_para.fileName       = [file_name_pre '_u1'];
visualizeImage(u1, [], visu_para); close all
visu_para.fileName       = [file_name_pre '_u2'];
visualizeImage(u2, [], visu_para); close all

visu_para_diff.fileName       = [file_name_pre '_u2-u1'];
visualizeImage(u2-u1, [], visu_para_diff); close all

visu_para_diff.fileName       = [folder '/OpticalFlowIllustration_dummyImage'];
visualizeImage(zeros(size(u2)), [], visu_para_diff); close all

dummyRGB = RGB_Lin(:,:,1) .* 0;
visu_para_diff.fileName       = [folder '/OpticalFlowIllustration_dummyV'];
visualizeImage(dummyRGB, [], visu_para_diff); close all



% warpImage(u2,  v)
visu_para.fileName       = [file_name_pre '_Warp(u2,vLin)'];
visualizeImage(warpImage(u2,  v_lin), [], visu_para); close all
visu_para.fileName       = [file_name_pre '_Warp(u2,vLin3)'];
visualizeImage(warpImage(u2,  v_lin_3warp), [], visu_para); close all
visu_para.fileName       = [file_name_pre '_Warp(u2,vLinSym)'];
visualizeImage(warpImage(u2,  v_lin_sym), [], visu_para); close all
visu_para.fileName       = [file_name_pre '_Warp(u2,vNoLinFast)'];
visualizeImage(warpImage(u2,  v_nonlin_fast), [], visu_para); close all
visu_para.fileName       = [file_name_pre '_Warp(u2,vNoLin)'];
visualizeImage(warpImage(u2,  v_nonlin), [], visu_para); close all
visu_para.fileName       = [file_name_pre '_Warp(u2,vNoLinSym)'];
visualizeImage(warpImage(u2,  v_nonlin_sym), [], visu_para); close all

% warpImage(u1, -v)
visu_para.fileName       = [file_name_pre '_Warp(u1,-vLin)'];
visualizeImage(warpImage(u1,  -v_lin), [], visu_para); close all
visu_para.fileName       = [file_name_pre '_Warp(u1,-vLin3)'];
visualizeImage(warpImage(u1,  -v_lin_3warp), [], visu_para); close all
visu_para.fileName       = [file_name_pre '_Warp(u1,-vLinSym)'];
visualizeImage(warpImage(u1,  -v_lin_sym), [], visu_para); close all
visu_para.fileName       = [file_name_pre '_Warp(u1,-vNoLinFast)'];
visualizeImage(warpImage(u1,  -v_nonlin_fast), [], visu_para); close all
visu_para.fileName       = [file_name_pre '_Warp(u1,-vNoLin)'];
visualizeImage(warpImage(u1,  -v_nonlin), [], visu_para); close all
visu_para.fileName       = [file_name_pre '_Warp(u1,-vNoLinSym)'];
visualizeImage(warpImage(u1,  -v_nonlin_sym), [], visu_para); close all

% motion
motion_visu_para.print = true;
motion_visu_para.fileName       = [file_name_pre '_vLin'];
visualizeMotion(v_lin,    info_dummy, motion_visu_para);
motion_visu_para.fileName       = [file_name_pre '_vLin3'];
visualizeMotion(v_lin_3warp,    info_dummy, motion_visu_para);
motion_visu_para.fileName       = [file_name_pre '_vLinSym'];
visualizeMotion(v_lin_sym, info_dummy, motion_visu_para);
motion_visu_para.fileName       = [file_name_pre '_vLinFast'];
visualizeMotion(v_nonlin_fast,    info_dummy, motion_visu_para);
motion_visu_para.fileName       = [file_name_pre '_vNoLin'];
visualizeMotion(v_nonlin,    info_dummy, motion_visu_para);
motion_visu_para.fileName       = [file_name_pre '_vNoLinSym'];
visualizeMotion(v_nonlin_sym, info_dummy, motion_visu_para);

% Warp(u2, v) - u1
visu_para_diff.fileName       = [file_name_pre '_Warp(u2,vLin)-u1'];
visualizeImage(warpImage(u2,  v_lin)-u1, [], visu_para_diff); close all
visu_para_diff.fileName       = [file_name_pre '_Warp(u2,vLin3)-u1'];
visualizeImage(warpImage(u2,  v_lin_3warp)-u1, [], visu_para_diff); close all
visu_para_diff.fileName       = [file_name_pre '_Warp(u2,vLinSym)-u1'];
visualizeImage(warpImage(u2,  v_lin_sym)-u1, [], visu_para_diff); close all
visu_para_diff.fileName       = [file_name_pre '_Warp(u2,vNoLinFast)-u1'];
visualizeImage(warpImage(u2,  v_nonlin_fast)-u1, [], visu_para_diff); close all
visu_para_diff.fileName       = [file_name_pre '_Warp(u2,vNoLin)-u1'];
visualizeImage(warpImage(u2,  v_nonlin)-u1, [], visu_para_diff); close all
visu_para_diff.fileName       = [file_name_pre '_Warp(u2,vNoLinSym)-u1'];
visualizeImage(warpImage(u2,  v_nonlin_sym)-u1, [], visu_para_diff); close all

% 'Warp(u2, vLin) - u1 + Warp(u1, -vLin) - u2'
visu_para_diff.fileName       = [file_name_pre '_Warp(u2,vLin)-u1+Warp(u1,-vLin)-u2'];
visualizeImage(warpImage(u2, v_lin)-u1 + warpImage(u1, -v_lin)-u2, [], visu_para_diff); close all
visu_para_diff.fileName       = [file_name_pre '_Warp(u2,vLin3)-u1+Warp(u1,-vLin3)-u2'];
visualizeImage(warpImage(u2, v_lin_3warp)-u1 + warpImage(u1, -v_lin_3warp)-u2, [], visu_para_diff); close all
visu_para_diff.fileName       = [file_name_pre '_Warp(u2,vLinSym)-u1+Warp(u1,-vLinSym)-u2'];
visualizeImage(warpImage(u2, v_lin_sym)-u1 + warpImage(u1, -v_lin_sym)-u2, [], visu_para_diff); close all
visu_para_diff.fileName       = [file_name_pre '_Warp(u2,vNoLinFast)-u1+Warp(u1,-vNoLinFast)-u2'];
visualizeImage(warpImage(u2, v_nonlin_fast)-u1 + warpImage(u1, -v_nonlin_fast)-u2, [], visu_para_diff); close all
visu_para_diff.fileName       = [file_name_pre '_Warp(u2,vNoLin)-u1+Warp(u1,-vNoLin)-u2'];
visualizeImage(warpImage(u2, v_nonlin)-u1 + warpImage(u1, -v_nonlin)-u2, [], visu_para_diff); close all
visu_para_diff.fileName       = [file_name_pre '_Warp(u2,vNoLinSym)-u1+Warp(u1,-vNoLinSym)-u2'];
visualizeImage(warpImage(u2, v_nonlin_sym)-u1 + warpImage(u1, -v_nonlin_sym)-u2, [], visu_para_diff); close all
