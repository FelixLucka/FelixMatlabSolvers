function res = projBoxConstraints(x, constraint, con_range)
% PROJECTBOXCONSTRAINTS is a projector type function that projects the input 
%   into a box [lb, ub]
%
% DESCRIPTION:
%  res =  projectBoxConstraints(x, 'none')
%  res =  projectBoxConstraints(x, 'range', [0, 1])
%
%  INPUT:
%   x          - values to project
%   constraint - 'none' for no constraints (for compartibility) 
%                'nonNegative' - for [0, Inf]
%                'range'        - for [lb, ub]
%
%  OPTIONAL INPUT:
%   con_range - vector in the form [lb, ub] to specify the box constraints
%
%  OUTPUTS:
%   res - struct containing the results in the form needed by the other
%   algorithms of the toolbox: 
%       'x'  - result of the projection
%       'Jx' - energy of projection, always 0 for projections
%
% ABOUT:
%   author          - Felix Lucka
%   date            - 16.03.2018
%   last update     - 27.10.2023
%
% See also 

switch constraint
    case 'none'
        res.x = x;
    case 'nonNegative'
        res.x = max(0, x);
    case 'range'
        res.x  = min(con_range(2), max(con_range(1), x));
    otherwise
        error('invalid constraint, choose ''none'', ''nonNegative'' or ''range''')
end

res.Jx = 0;

end