function res = proxL21(Z, lambda, l2dims)
% PROXL21 applies the vectorial softhtresholding operator
%
% DESCRIBTION:
%   proxL21 applies the vectorial softthreholding operator which is solving
%   the problem
%   min_X L21norm(X) + 1/(2*lambda) || X - Z ||^2_Fro,
%   where L21norm is the L21 matrix norm which is computed by taking the L2 norm
%   along the l2dims dimensions on X followed by the sum over all remaining
%   dimensions 
%
% INPUT:
%   Z      - a multi-dimensional field
%   lambda - regularization parameter, can be a field (used in auto-expand)
%   l2dims - dimensions along which to use the L2 norm
%   
%
%  OUTPUTS:
%   res - struct containing the results in the form needed by the other
%   algorithms of the toolbox: 
%       'x'  - result of the softthresholding
%       'Jx' - L21norm(x)
%
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 05.04.2018
%       last update     - 27.10.2023
%
% See also


% compute the L2 norm along the l2 dimensions

l2_norm = Z.^2;
for i_dim = l2dims
    l2_norm = sum(l2_norm, i_dim);
end
l2_norm = sqrt(l2_norm);


% get the positions of the non-zeros
loc_non_zeros = l2_norm > 0;

% compute the factor 
factor = max(bsxfun(@minus, l2_norm, lambda), 0);
factor(loc_non_zeros) = factor(loc_non_zeros)./l2_norm(loc_non_zeros);

% multiply the X with the factor 
res.x  = bsxfun(@times, factor, Z);

l2_norm = res.x.^2;
for i_dim = l2dims
    l2_norm = sum(l2_norm, i_dim);
end
l2_norm = sqrt(l2_norm);
res.Jx = sumAll(sqrt(l2_norm));

end