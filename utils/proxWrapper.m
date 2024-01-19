function res = proxWrapper(name, z, lambda, para, update)
%PROXWRAPPER is a wrapper to turn denosing functions into prox operators
%
% DESCRIPTION: 
%   proxWrapper.m can be used to convert denoising functions into prox operators, 
%   i.e., functions that solve 
%   argmin_x ( G(x) + 1/(2*lambda) || x - z ||_2^2   )
%   for a functional G. The function returns a struct with the field 'x', 
%   which is the minimizer of the above functional but may also contain other fields 
%   (e.g. for functionals that are defined as J(z) = argmin_w R(z,w), w and J(z) 
%
% USAGE:
%   res = proxWrapper('TV', z, alpha, previousRes, para)
%
% INPUTS:
%   name   - name of the denoising function
%   z      - image to denoise+
%   lambda - positive scalar, see above
%
% OPTIONAL INPUTS:
%   para - a struct mostly containing parameters for the denosing routine. 
%          The extra field 'warmstart' contains a locial indicating whether
%          it should be update by the fields in update. 
%   update - part of the result struct of the last call to the denosing function. Any
%            fields in para that also exist in update will be overwritten with the
%            ones from update
%
% OUTPUTS:
%   res - struct containing the results in the form needed by the other
%   algorithms of the toolbox: 
%       'x'  - result of the softthresholding
%       'Jx' - L21norm(x)
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 10.04.2018
%       last update     - 10.04.2018
%
% See also

% check user defined value for res, otherwise assign default value
if(nargin < 4)
    para = [];
end

% check user defined value for para, otherwise assign default value
if(nargin < 5)
    update = [];
end

% update fields of para that are also in res
warmstart = checkSetInput(para, 'warmstart', 'logical', false);
if(warmstart)
    para = overwriteFields(para, update, true);
end

% call desnoing function
switch name
    case 'TV'
        [res.x, res.update.y, res.iter, info]  = TVdenoising(z, lambda, para);
        res.update.x = res.x;
        res.Jx = info.Jx;
    otherwise
        notImpErr
end


end