function du_dt = tempFwdDiff(u, adjoint_fl, dt, dim_u)
%TEMPORALFWDGRAD computes the forward finite difference in time or its
%adjoint operation
%   ut = temporalFwdGrad(u, true)
%   u  = temporalFwdGrad(ut,false)

%  INPUTS:
%   u - 3D or 4D numerical array. The last dimension is assumped to
%   correspond to the time dimenions and the first ones to the spatial ones
%   adjointFL - boolean that idicates whether the forward or the adjoint
%   transport operator should be applied.
%
% OPTIONAL INPUTS
%   dt - the time differences between frames as a vector
%       (size T-1 x 1) which will be used to weight the temporal finite
%       differences.
%
%  OUTPUTS:
%   du_dt - temporal forward differences of u or output of adjoint operation
%
% ABOUT:
%       author          - Felix Lucka
%       date            - ??.??.2017
%       last update     - 27.10.2023
%
% See also

if(nargin < 3 || isempty(dt) || isequal(dt, 1))
    dt_weight = false;
else
    dt_weight = true;
end

if(nargin < 4)
    dim_u = nDims(u) - 1;
end

if(isequal(nDims(u), dim_u))
    sz_u  = size(u);
    n_t   = 1;
else
    sz_u  = size(u);
    n_t   = sz_u(end);
    sz_u  = sz_u(1:end-1);
end

if(~adjoint_fl)
    
    if(any(u(:)))
        % apply forward finite difference in time
        du_dt = diff(u, 1, dim_u+1);
        if(dt_weight)
            switch dim_u
                case 2
                    du_dt = bsxfun(@times, du_dt, 1./ reshape(dt(:), 1, 1, []));
                case 3
                    du_dt = bsxfun(@times, du_dt, 1./ reshape(dt(:), 1, 1, 1, []));
                otherwise
                    notImpErr
            end
        end
    else
        du_dt  = zeros([sz_u, n_t-1],'like',u);
    end
else
    
    n_t = n_t + 1;
    du_dt  = zeros([sz_u, n_t],'like',u);
    if(any(u(:)))
        % apply adjoint finite difference in time
        switch dim_u
            case 2 % we are in 2D
                if(dt_weight)
                    u = bsxfun(@times, u, 1./ reshape(dt(:), 1, 1, []));
                end
                du_dt(:,:,1)       = -u(:, :, 1);
                du_dt(:,:,2:end-1) = -diff(u, 1, 3);
                du_dt(:,:,end)     =  u(:, :, end);
            case 3 % we are in 3D
                if(dt_weight)
                    u = bsxfun(@times, u, 1./ reshape(dt(:), 1, 1, 1, []));
                end
                du_dt(:,:,:,1) = -u(:,:,:,1);
                du_dt(:,:,:,2:end-1) = -diff(u,1,4);
                du_dt(:,:,:,end) = u(:,:,:,end);
            otherwise
                notImpErr
        end
    end
    
end