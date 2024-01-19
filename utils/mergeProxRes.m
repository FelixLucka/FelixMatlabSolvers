function res = mergeProxRes(varargin)
%MERGEPROXRES merges the result structs of different proximal operators
%
% DESCRIPTION: 
%   functionTemplate.m can be used as a template 
%
% USAGE:
%   res = mergeProxRes(res1, res2, res3)
%
% INPUTS:
%   res - result structs of the application of proximal operators
%
% OUTPUTS:
%   res - a struct with the same fields as the individual input structs but 
%         as cell arrays containing the individual fields of all input
%         struts
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 10.04.2018
%       last update     - 21.12.2023
%
% See also

res    = [];
res.x  = {};
res.Jx = {};

for i_res = 1:nargin
   res.x{i_res}  = varargin{i_res}.x; 
   res.Jx{i_res} = varargin{i_res}.Jx; 
end

end