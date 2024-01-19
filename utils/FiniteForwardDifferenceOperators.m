function [D, DT] = FiniteForwardDifferenceOperators(dim, BC)
% FINITEDIFFERENCEOPERATORS returns function handles to finite forward difference operators 
% along different dimensions and their adjoints
%
% DESCRIBTION:
%   FiniteDifferenceOperators returns function handles to finite forward difference operators 
%   along different dimensions and their adjoints. Different boundary conditions 
%   can be applied for each dimension. 
%
%  INPUT:
%   dim - dimension of iput array
%   BC  - a cell of strings specifying the boundary conditions used
%
%  OUTPUTS:
%   D  - cell of function handles for the finite difference operators along
%        each dimension
%   DT - cell of function handles for the corresponding adjoints
%
% ABOUT:
%   author          - Felix Lucka
%   date            - 16.03.2018
%   last update     - 27.10.2023
%
% See also 

if(ischar(BC))
   BC = repIntoCell(BC, [dim, 1]);
end

D  = {};
DT = {};
ones_dim = ones(1, dim);

for iDim = 1:dim
   switch BC{iDim}
       case  '0' % zero boundary conditions
           aux       = ones_dim;
           aux(iDim) = 0;
           D{iDim}  = @(x) padarray(cat(iDim, sliceArray(x, iDim, 1), diff(x, 1, iDim), - sliceArray(x, iDim, size(x, iDim))), aux, 0 ,'pre');
           DT{iDim} = @(x) cutArray(- diff(x, 1, iDim), aux, 'pre');
        case {'none','NB'} % Neuman boundary conditions
           D{iDim}  = @(x) cat(iDim, diff(x, 1, iDim), 0 * sliceArray(x, iDim, 1));
           DT{iDim} = @(x) cat(iDim, -sliceArray(x, iDim, 1), -diff(sliceArray(x, iDim, 1:size(x, iDim)-1), 1, iDim), sliceArray(x, iDim, size(x, iDim)-1));    
       otherwise
           notImpErr
   end
end

end
