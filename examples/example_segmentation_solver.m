%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script demonstrates the use of the function smoothTV_Segmentation.m
% for segmentation using smooth TV contraints
%
% USAGE: Make sure that FelixMatlabTools and FelixMatlabSolvers are on the path!
%
% ABOUT:
% 	author          - Felix Lucka
% 	date            - 21.12.2023
%  	last update     - 21.12.2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

addpath(genpath('../FelixMatlabTools'))
addpath(genpath(pwd))

%% segmentation with TV constraints
rng(1)

n_x = 128;

% make ground trugh segmentation 
seg    = makeSphere([n_x,n_x],[0,1;0,1], [0.7,0.7], 0.15, 1);
seg    = cat(3, seg, makeRectangle([n_x,n_x], [0,1;0,1], [0.25,0.25], [0.2,0.2], 1));
seg    = cat(3, 1 - sum(seg, 3), seg);

% generate noisy data 
c          = [0, 0.5, 1];
sigma      = 0.1;
u_true      = sum(bsxfun(@times, seg, reshape(c, 1, 1, 3)), 3);
f_noisy     = u_true + sigma * randn(size(u_true));

% compute the likelihood weights that enter the segmentation algorithm
weights    = (f_noisy - reshape(c, [1,1,3])).^2 / (2*sigma^2);
weights    = weights / max(weights(:));

% set up the parameters for the optimization function 
opt_para         = [];
opt_para.maxIter = 1000; 
opt_para.epsilon = 10^-3;
opt_para.output  = true;
alpha            = 10^0;

[u, info]  = smoothTV_Segmentation(weights, alpha, opt_para);

subplot(1,3,1); imagesc(f_noisy);
subplot(1,3,2); image(getfield(projProbabilities(1./weights), 'x'));
subplot(1,3,3); image(u);

