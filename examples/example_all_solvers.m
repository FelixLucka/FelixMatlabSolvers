%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script demonstrates the use of some of the solvers implemented in
% the toolbox (so far, mostly TV reguarlization, but I will extend it)
%
% USAGE: Make sure that FelixMatlabTools and FelixMatlabSolvers are on the path!
%
% ABOUT:
% 	author          - Felix Lucka
% 	date            - ??.??.20??
%  	last update     - 19.01.2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%% TV denoising routines

% shep logan phantom
u_true  = phantom(128);
f_noisy = u_true + 0.1 * randn(size(u_true));
clim   = [min(f_noisy(:)), max(f_noisy(:))];
alpha  = 0.1;

f_TV     = {};
info_TV  = {};

% default settings
[f_TV{end+1}, ~, ~, info_TV{end+1}]  = TV_Denoising(f_noisy, alpha);

% denoise only along first coordinate
para = [];
para.coordinates = 1;
[f_TV{end+1}, ~, ~, info_TV{end+1}]  = TV_Denoising(f_noisy, alpha, para);

% range constraint 
para = [];
para.constraint = 'range';
para.conRange   = [0, 0.5];
[f_TV{end+1}, ~, ~, info_TV{end+1}]  = TV_Denoising(f_noisy, alpha, para);

% switch acceleration off
para = [];
para.acceleration = false;
[f_TV{end+1}, ~, ~, info_TV{end+1}]  = TV_Denoising(f_noisy, alpha, para);

% switch acceleration off and try different theta
para = [];
para.acceleration = false;
para.theta = 0.5;
[f_TV{end+1}, ~, ~, info_TV{end+1}]  = TV_Denoising(f_noisy, alpha, para);

% split in two
para = [];
para.acceleration = false;
para.returnMinEnergy = false;
para.maxIter      = 50;
[para.x, para.y, ~, info]            = TV_Denoising(f_noisy, alpha, para);
[f_TV{end+1}, ~, ~, info_TV{end+1}]  = TV_Denoising(f_noisy, alpha, para);
% concat energies
info_TV{end}.energy = [info.energy; info_TV{end}.energy(2:end)];
info_TV{end}.iter   = info_TV{end}.iter + info.iter;

% test prox operator handle
para = [];
para.constraint = 'range';
para.conRange   = [0, 0.5];
para.acceleration         = false;
para.uniConvexFunctional  = 'F*';
para.convexityModulus     = 1;
para.returnMinEnergy      = false;
Id = @(x) x;
[f_TV{end+1}, ~, ~, info_TV{end+1}]  = TV_Deblurring(Id, Id, f_noisy, alpha, para);
 
n_rec = length(f_TV);

figure();
subplot(2, n_rec, 1); imagesc(u_true, clim)
subplot(2, n_rec, 2); imagesc(f_noisy, clim)

for i=1:n_rec
    subplot(2, n_rec, n_rec+i); imagesc(f_TV{i}, clim)
end

figure(); 
for i=1:n_rec
    plot(0:info_TV{i}.iter, info_TV{i}.energy); hold on
end

%% smoothed TV denoising

tmp_para         = [];
tmp_para.maxEval = 10^3;
tmp_para.output  = true;
tmp_para.functional = 'Huber';
tmp_para.epsilon    = 10^-2 * max(u_true(:));

f_TV_sm     = {};
info_TV_sm  = {};


% default settings
para = tmp_para;
[f_TV_sm{end+1},info_TV_sm{end+1}]  = smoothTV_Deblurring(Id, Id, f_noisy, alpha, para);


% sharpen epsilon
para            = tmp_para;
para.epsilon    = para.epsilon/10;
[f_TV_sm{end+1},info_TV_sm{end+1}]  = smoothTV_Deblurring(Id, Id, f_noisy, alpha, para);


% switch functional
para               = tmp_para;
para.functional = 'SqrtEpsilon';

[f_TV_sm{end+1},info_TV_sm{end+1}]  = smoothTV_Deblurring(Id, Id, f_noisy, alpha, para);

% sharpen epsilon
para.epsilon = para.epsilon / 10;
[f_TV_sm{end+1},info_TV_sm{end+1}]  = smoothTV_Deblurring(Id, Id, f_noisy, alpha, para);


n_rec = length(f_TV_sm);

figure();
subplot(2, n_rec, 1); imagesc(u_true, clim)
subplot(2, n_rec, 2); imagesc(f_noisy, clim)
subplot(2, n_rec, 3); imagesc(f_TV{1}, clim)

for i=1:n_rec
    subplot(2, n_rec, n_rec+i); imagesc(f_TV_sm{i}, clim)
end

figure(); 
for i=1:n_rec
    plot(info_TV_sm{i}.FGradEvalVec, info_TV_sm{i}.energy); hold on
end


% error to TV solution
for i=1:n_rec
    norm(f_TV{1}(:) - f_TV_sm{i}(:)) / norm(f_TV{1}(:))
end



%% TV deblurring routines

%close all

tmp_para         = [];
tmp_para.maxIter = 1000;
tmp_para.output  = true;

Aker = fspecial('gaussian', [13, 13], 1.5);
A    = @(x) imfilter(x, Aker, 'circular');
Aadj = @(x) A(x);

% shep logan phantom
u_true  = phantom(128);
f_noisy = A(u_true) + 0.1 * randn(size(u_true));
clim   = [min(u_true(:)), max(u_true(:))];
alpha  = 0.1;

f_TV     = {};
info_TV  = {};

% default settings
[f_TV{end+1}, ~, ~, info_TV{end+1}]  = TV_Deblurring(A, Aadj, f_noisy, alpha, tmp_para);

% different theta parameter
para = tmp_para;
para.theta = 0.5;
[f_TV{end+1}, ~, ~, info_TV{end+1}]  = TV_Deblurring(A, Aadj, f_noisy, alpha, para);

% range constraint 
para = tmp_para;
para.constraint = 'range';
para.conRange   = [0, 0.5];
[f_TV{end+1}, ~, ~, info_TV{end+1}]  = TV_Deblurring(A, Aadj, f_noisy, alpha, para);

% switch acceleration on
para = tmp_para;
para.acceleration         = true;
para.uniConvexFunctional  = 'F*';
para.convexityModulus     = 1;
[f_TV{end+1}, ~, ~, info_TV{end+1}]  = TV_Deblurring(A, Aadj, f_noisy, alpha, para);

n_rec = length(f_TV);

figure();
subplot(2, n_rec, 1); imagesc(u_true, clim)
subplot(2, n_rec, 2); imagesc(f_noisy, clim)

for i=1:n_rec
    subplot(2, n_rec, n_rec+i); imagesc(f_TV{i}, clim)
end

figure(); 
for i=1:n_rec
    plot(0:info_TV{i}.iter, info_TV{i}.energy); hold on
end


%% segmentation with TV constraints
rng(1)

% project onto probability simplex
u_true      = phantom(128);
background = u_true < 0.05;
skin       = u_true > 0.5;
soft       = (0.5 > u_true) & (u_true > 0.05);

seg        = cat(3, background, skin, soft);
c          = [0, 0.5, 1];
sigma      = 0.1;
u_true      = sum(bsxfun(@times, seg, reshape(c, 1, 1, 3)), 3);
f_noisy     = u_true + sigma * randn(size(u_true));

weights    = (f_noisy - reshape(c, [1,1,3])).^2 / (2*sigma^2);
weights    = weights / max(weights(:));


opt_para         = [];
opt_para.maxIter = 1000; 
opt_para.epsilon = 10^-3;
opt_para.output  = true;
alpha           = 10^0;

[u, info]  = smoothTV_Segmentation(weights, alpha, opt_para);


subplot(1,3,1); imagesc(f_noisy);
subplot(1,3,2); image(getfield(projProbabilities(1./weights), 'x'));
subplot(1,3,3); image(u);

