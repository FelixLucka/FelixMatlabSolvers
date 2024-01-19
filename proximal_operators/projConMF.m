function [P, info] = projConMF(Q, rank, pro_U, proV, con_para)
% PROJECTORCONMF factorizes a matrix Q by conMatrixFactorization.m and
% just returns the product of the factorization (as a projection of Q)
%
%  P = projectorConMF(Q,rank,pro_U,pro_V,conPara)
%
%  INPUT:
%   see conMatrixFactorization.m
%
%  OUTPUTS:
%   P - U*V
%   info - a struct summarizing some info about the factorization
%
%
% ABOUT:
%   author          - Felix Lucka
%   date            - ??.??.????
%   last update     - 27.10.2023
%
% See also 


[U, V] = conMatrixFactorization(Q, rank, pro_U, proV, con_para);
P     = U*V;

info.residualNorm = norm(Q - P,'fro');

end