function res = dynamicOperator(A, x, cell_output) 
% DYNAMICOPERTOR applies a dynamic operator or its adjoint to a sequences of 
% inputs
%
% DESCRIBTION:
%  dynamicOperator is a wrapper that takes a cell of operators and applies
%  them to an input or cell of inputs
%
% INPUTS:
%   A: cell or function handles to the forward operator, it is
%      assumed to repeat periodically
%   x: cell or numerical array of inputs. In the later case, the last
%      dimension is considered the temporal dimension.
%
% OPTIONAL INPUTS:
%   cell_output: bool controling whether the function should return a cell
%   in any case (default: false)
%
% OUTPUTS:
%   res - output in the same format as input (cell or array)
%
% ABOUT:
%   author          - Felix Lucka
%   date            - 06.05.2018
%   last update     - 27.10.2023
%
% See also

% check user defined value for cellOutput, otherwise assign default value
if(nargin < 3)
    cell_output = false;
end

% check if A is a cell and convert otherwise
if(~iscell(A))
    A = {A};
end
period_length_A = length(A);
t2p             = @(t) mod(t - 1, period_length_A) + 1;

% check in format x comes and how many time frames it has
is_cell_x = iscell(x);
if(is_cell_x)
    T   = length(x);
else
    dim_time = nDims(x);
    T        = size(x, dim_time);
end
res = cell(1, T);


for t = 1:T
    if(is_cell_x)
        res{t} = A{t2p(t)}(x{t});
    else
        res{t} = A{t2p(t)}(sliceArray(x, dim_time, t, true));
    end
end

if(~cell_output && ~is_cell_x)
    res = cat(nDims(res{t})+1, res{:});
end

end
