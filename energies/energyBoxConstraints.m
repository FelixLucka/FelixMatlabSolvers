function energy = energyBoxConstraints(x, constraint, con_range)
% PROJECTBOXCONSTRAINTS is a energy type function that returns 0 of the input 
%   fullfills the box constraints and infinity otherwise
%
% DESCRIPTION:
%  energy = energyBoxConstraints(x, 'non-negative')
%
%  INPUT:
%   x          - object to test
%   constraint - 'none' for no constraints (for compartibility) 
%                'non-negative' - for [0, Inf]
%                'range'        - for [lb, ub]
%
%  OPTIONAL INPUT:
%   con_range - vector in the form [lb, ub] to specify the box constraints
%
%  OUTPUTS:
%   energy - 0 of the input fullfills the box constraints and infinity otherwise
%
% ABOUT:
%   author          - Felix Lucka
%   date            - 06.05.2018
%   last update     - 26.10.2023
%
% See also 

switch constraint
    case 'none'
        energy = 0;
    case 'nonNegative'
        energy = (1./not(any(x(:) < 0))-1);
    case 'range'
        energy = (1./not(any(x(:) < con_range(1)))-1)  + (1./not(any(x(:) > con_range(2)))-1);
    otherwise
        error('invalid constraint, choose ''none'', ''nonNegative'' or ''range''')
end

end