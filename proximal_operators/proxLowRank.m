function X = proxLowRank(X, lambda)
% PROXLOWRANK solves
% Z = argmin_Z {NN(Z) + 1/(2*lambda) \|X - Z\|_{Fro} }
% where NN(Z) is the nuclear norm of the matrix Z
%
%  INPUT:
%   X - matrix
%   lambda - regularization parameter
%
%  OUTPUTS:
%   X - solution of the proximal problem
%
% ABOUT:
%       author          - Felix Lucka
%       date            - ??.??.????
%       last update     - 27.10.2023
%
% See also prox*.m

% compute SVD of X
[U,S,V] = svd(X,'econ');
% apply soft thresholding to S
S = sign(S) .* max(0,abs(S) - lambda);
% construct Z
X = U*S*V';

end