function y = spatialFwdJacobian(x, adjoint_fl, single_frame_fl)
% SPATIALFWDJACOBIAN computes the spatial jacobian of a spatio-temporal 
% vector field in 2D or 3D or its adjoint
%   Jv = spatialFwdJacobian(v,false,singleFrameFL)

%  INPUTS:
%   x - 3D, 4D or 5D numerical array. If
%       singleFrameFL = true:  [n_x,n_y,2] or [n_x,n_y,n_z,3]
%       singleFrameFL = false: [n_x,n_y,2,n_t] or [n_x,n_y,n_z,3,n_t]
%   adjointFL - boolean that idicates whether the forward or the adjoint
%   spatial jacobian operator should be applied. 
%   singleFrameFL - boolean that idicates whether the input has no temporal
%   dimension
%           
%
%  OUTPUTS:
%   y - spatial forward Jacobian. If
%       singleFrameFL = true:  [n_x,n_y,2,2] or [n_x,n_y,n_z,3,3]
%       singleFrameFL = false: [n_x,n_y,2,2,n_t] or [n_x,n_y,n_z,3,3,n_t]
%
% ABOUT:
%       author          - Felix Lucka
%       date            - ??.??.2017
%       last update     - 27.10.2023
%
% See also

if(nargin < 3)
    single_frame_fl = false;
end

if(~single_frame_fl)
    
    sz_x = size(x);
    n_t  = sz_x(end);
    
    if(~adjoint_fl)
        dim_space = ndims(x)-2;
        n_xyz     = sz_x(1:dim_space);
        y = zeros([n_xyz, dim_space, dim_space, n_t],'like',x);
        if(~any(x(:))); return; end
        switch dim_space
            case 2
                % spatial derivatives of the x component of v
                y(1:end-1,:,1,1,:) = diff(x(:,:,1,:),1,1);
                y(:,1:end-1,1,2,:) = diff(x(:,:,1,:),1,2);
                % spatial derivatives of the y component of v
                y(1:end-1,:,2,1,:) = diff(x(:,:,2,:),1,1);
                y(:,1:end-1,2,2,:) = diff(x(:,:,2,:),1,2);
            case 3
                % spatial derivatives of the x component of v
                y(1:end-1,:,:,1,1,:) = diff(x(:,:,:,1,:),1,1);
                y(:,1:end-1,:,1,2,:) = diff(x(:,:,:,1,:),1,2);
                y(:,:,1:end-1,1,3,:) = diff(x(:,:,:,1,:),1,3);
                % spatial derivatives of the y component of v
                y(1:end-1,:,:,2,1,:) = diff(x(:,:,:,2,:),1,1);
                y(:,1:end-1,:,2,2,:) = diff(x(:,:,:,2,:),1,2);
                y(:,:,1:end-1,2,3,:) = diff(x(:,:,:,2,:),1,3);
                % spatial derivatives of the z component of v
                y(1:end-1,:,:,3,1,:) = diff(x(:,:,:,3,:),1,1);
                y(:,1:end-1,:,3,2,:) = diff(x(:,:,:,3,:),1,2);
                y(:,:,1:end-1,3,3,:) = diff(x(:,:,:,3,:),1,3);
            otherwise
                notImpErr
        end
    else
        dim_space = ndims(x)-3;
        n_xyz     = sz_x(1:dim_space);
        sz_y      = [n_xyz, dim_space, n_t];
        y         = zeros(sz_y,'like',x);
        dummy_y   = zeros([n_xyz n_t],'like',x);
        if(~any(x(:))); return; end
        switch dim_space
            case 2
                % fwDX'*X(1,1)
                dummy_y(1,:,:) = -x(1,:,1,1,:);
                dummy_y(2:end-1,:,:) = -diff(x(1:end-1,:,1,1,:),1,1);
                dummy_y(end,:,:) = x(end-1,:,1,1,:);
                y(:,:,1,:) = dummy_y;
                % fwDY'*X(1,2)
                dummy_y(:,1,:) = -x(:,1,1,2,:);
                dummy_y(:,2:end-1,:) = -diff(x(:,1:end-1,1,2,:),1,2);
                dummy_y(:,end,:) = x(:,end-1,1,2,:);
                y(:,:,1,:) = y(:,:,1,:) + dummy_y;
                % fwDX'*X(2,1)
                dummy_y(1,:,:) = -x(1,:,2,1,:);
                dummy_y(2:end-1,:,:) = -diff(x(1:end-1,:,2,1,:),1,1);
                dummy_y(end,:,:) = x(end-1,:,2,1,:);
                y(:,:,2,:) = dummy_y;
                % fwDY'*X(2,2)
                dummy_y(:,1,:) = -x(:,1,2,2,:);
                dummy_y(:,2:end-1,:) = -diff(x(:,1:end-1,2,2,:),1,2);
                dummy_y(:,end,:) = x(:,end-1,2,2,:);
                y(:,:,2,:) = y(:,:,2,:) + dummy_y;
            case 3
                % fwDX'*X(1,1)
                dummy_y(1,:,:,:) = -x(1,:,:,1,1,:);
                dummy_y(2:end-1,:,:,:) = -diff(x(1:end-1,:,:,1,1,:),1,1);
                dummy_y(end,:,:,:) = x(end-1,:,:,1,1,:);
                y(:,:,:,1,:) = dummy_y;
                % fwDY'*X(1,2)
                dummy_y(:,1,:,:) = -x(:,1,:,1,2,:);
                dummy_y(:,2:end-1,:,:) = -diff(x(:,1:end-1,:,1,2,:),1,2);
                dummy_y(:,end,:,:) = x(:,end-1,:,1,2,:);
                y(:,:,:,1,:) = y(:,:,:,1,:) + dummy_y;
                % fwDZ'*X(1,3)
                dummy_y(:,:,1,:) = -x(:,:,1,1,3,:);
                dummy_y(:,:,2:end-1,:) = -diff(x(:,:,1:end-1,1,3,:),1,3);
                dummy_y(:,:,end,:) = x(:,:,end-1,1,3,:);
                y(:,:,:,1,:) = y(:,:,:,1,:) + dummy_y;
                % fwDX'*X(2,1)
                dummy_y(1,:,:,:) = -x(1,:,:,2,1,:);
                dummy_y(2:end-1,:,:,:) = -diff(x(1:end-1,:,:,2,1,:),1,1);
                dummy_y(end,:,:,:) = x(end-1,:,:,2,1,:);
                y(:,:,:,2,:) = dummy_y;
                % fwDY'*X(2,2)
                dummy_y(:,1,:,:) = -x(:,1,:,2,2,:);
                dummy_y(:,2:end-1,:,:) = -diff(x(:,1:end-1,:,2,2,:),1,2);
                dummy_y(:,end,:,:) = x(:,end-1,:,2,2,:);
                y(:,:,:,2,:) = y(:,:,:,2,:) + dummy_y;
                % fwDZ'*X(2,3)
                dummy_y(:,:,1,:) = -x(:,:,1,2,3,:);
                dummy_y(:,:,2:end-1,:) = -diff(x(:,:,1:end-1,2,3,:),1,3);
                dummy_y(:,:,end,:) = x(:,:,end-1,2,3,:);
                y(:,:,:,2,:) = y(:,:,:,2,:) + dummy_y;
                % fwDX'*X(3,1)
                dummy_y(1,:,:,:) = -x(1,:,:,3,1,:);
                dummy_y(2:end-1,:,:,:) = -diff(x(1:end-1,:,:,3,1,:),1,1);
                dummy_y(end,:,:,:) = x(end-1,:,:,3,1,:);
                y(:,:,:,3,:) = dummy_y;
                % fwDY'*X(3,2)
                dummy_y(:,1,:,:) = -x(:,1,:,3,2,:);
                dummy_y(:,2:end-1,:,:) = -diff(x(:,1:end-1,:,3,2,:),1,2);
                dummy_y(:,end,:,:) = x(:,end-1,:,3,2,:);
                y(:,:,:,3,:) = y(:,:,:,3,:) + dummy_y;
                % fwDZ'*X3,3)
                dummy_y(:,:,1,:) = -x(:,:,1,3,3,:);
                dummy_y(:,:,2:end-1,:) = -diff(x(:,:,1:end-1,3,3,:),1,3);
                dummy_y(:,:,end,:) = x(:,:,end-1,3,3,:);
                y(:,:,:,3,:) = y(:,:,:,3,:) + dummy_y;
            otherwise
                notImpErr
        end
    end
else
    if(~adjoint_fl)
        dim_space = ndims(x)-1;
        y = zeros([size(x),dim_space],'like',x);
        if(~any(x(:))); return; end
        switch dim_space
            case 2
                % spatial derivatives of the x component of v
                y(1:end-1,:,1,1) = diff(x(:,:,1),1,1);
                y(:,1:end-1,1,2) = diff(x(:,:,1),1,2);
                % spatial derivatives of the y component of v
                y(1:end-1,:,2,1) = diff(x(:,:,2),1,1);
                y(:,1:end-1,2,2) = diff(x(:,:,2),1,2);
            case 3
                % spatial derivatives of the x component of v
                y(1:end-1,:,:,1,1) = diff(x(:,:,:,1),1,1);
                y(:,1:end-1,:,1,2) = diff(x(:,:,:,1),1,2);
                y(:,:,1:end-1,1,3) = diff(x(:,:,:,1),1,3);
                % spatial derivatives of the y component of v
                y(1:end-1,:,:,2,1) = diff(x(:,:,:,2),1,1);
                y(:,1:end-1,:,2,2) = diff(x(:,:,:,2),1,2);
                y(:,:,1:end-1,2,3) = diff(x(:,:,:,2),1,3);
                % spatial derivatives of the z component of v
                y(1:end-1,:,:,3,1) = diff(x(:,:,:,3),1,1);
                y(:,1:end-1,:,3,2) = diff(x(:,:,:,3),1,2);
                y(:,:,1:end-1,3,3) = diff(x(:,:,:,3),1,3);
            otherwise
                notImpErr
        end
    else
        dim_space = ndims(x)-2;
        sz_x = size(x);
        y = zeros(sz_x(1:length(sz_x)-1),'like',x);
        if(~any(x(:))); return; end
        switch dim_space
            case 2
                % fwDX'*X(1,1)
                dummy_y = zeros(sz_x(1:2),'like',x);
                dummy_y(1,:) = -x(1,:,1,1);
                dummy_y(2:end-1,:) = -diff(x(1:end-1,:,1,1),1,1);
                dummy_y(end,:) = x(end-1,:,1,1);
                y(:,:,1) = dummy_y;
                % fwDY'*X(1,2)
                dummy_y(:,1) = -x(:,1,1,2);
                dummy_y(:,2:end-1) = -diff(x(:,1:end-1,1,2),1,2);
                dummy_y(:,end) = x(:,end-1,1,2);
                y(:,:,1) = y(:,:,1) + dummy_y;
                % fwDX'*X(2,1)
                dummy_y(1,:) = -x(1,:,2,1);
                dummy_y(2:end-1,:) = -diff(x(1:end-1,:,2,1),1,1);
                dummy_y(end,:) = x(end-1,:,2,1);
                y(:,:,2) = dummy_y;
                % fwDY'*X(2,2)
                dummy_y(:,1) = -x(:,1,2,2);
                dummy_y(:,2:end-1) = -diff(x(:,1:end-1,2,2),1,2);
                dummy_y(:,end) = x(:,end-1,2,2);
                y(:,:,2) = y(:,:,2) + dummy_y;
            case 3
                % fwDX'*X(1,1)
                dummy_y = zeros(sz_x(1:3),'like',x);
                dummy_y(1,:,:) = -x(1,:,:,1,1);
                dummy_y(2:end-1,:,:) = -diff(x(1:end-1,:,:,1,1),1,1);
                dummy_y(end,:,:) = x(end-1,:,:,1,1);
                y(:,:,:,1) = dummy_y;
                % fwDY'*X(1,2)
                dummy_y(:,1,:) = -x(:,1,:,1,2);
                dummy_y(:,2:end-1,:) = -diff(x(:,1:end-1,:,1,2),1,2);
                dummy_y(:,end,:) = x(:,end-1,:,1,2);
                y(:,:,:,1) = y(:,:,:,1) + dummy_y;
                % fwDZ'*X(1,3)
                dummy_y(:,:,1) = -x(:,:,1,1,3);
                dummy_y(:,:,2:end-1) = -diff(x(:,:,1:end-1,1,3),1,3);
                dummy_y(:,:,end) = x(:,:,end-1,1,3);
                y(:,:,:,1) = y(:,:,:,1) + dummy_y;
                % fwDX'*X(2,1)
                dummy_y(1,:,:) = -x(1,:,:,2,1);
                dummy_y(2:end-1,:,:) = -diff(x(1:end-1,:,:,2,1),1,1);
                dummy_y(end,:,:) = x(end-1,:,:,2,1);
                y(:,:,:,2) = dummy_y;
                % fwDY'*X(2,2)
                dummy_y(:,1,:) = -x(:,1,:,2,2);
                dummy_y(:,2:end-1,:) = -diff(x(:,1:end-1,:,2,2),1,2);
                dummy_y(:,end,:) = x(:,end-1,:,2,2);
                y(:,:,:,2) = y(:,:,:,2) + dummy_y;
                % fwDZ'*X(2,3)
                dummy_y(:,:,1) = -x(:,:,1,2,3);
                dummy_y(:,:,2:end-1) = -diff(x(:,:,1:end-1,2,3),1,3);
                dummy_y(:,:,end) = x(:,:,end-1,2,3);
                y(:,:,:,2) = y(:,:,:,2) + dummy_y;
                % fwDX'*X(3,1)
                dummy_y(1,:,:) = -x(1,:,:,3,1);
                dummy_y(2:end-1,:,:) = -diff(x(1:end-1,:,:,3,1),1,1);
                dummy_y(end,:,:) = x(end-1,:,:,3,1);
                y(:,:,:,3) = dummy_y;
                % fwDY'*X(3,2)
                dummy_y(:,1,:) = -x(:,1,:,3,2);
                dummy_y(:,2:end-1,:) = -diff(x(:,1:end-1,:,3,2),1,2);
                dummy_y(:,end,:) = x(:,end-1,:,3,2);
                y(:,:,:,3) = y(:,:,:,3) + dummy_y;
                % fwDZ'*X3,3)
                dummy_y(:,:,1) = -x(:,:,1,3,3);
                dummy_y(:,:,2:end-1) = -diff(x(:,:,1:end-1,3,3),1,3);
                dummy_y(:,:,end) = x(:,:,end-1,3,3);
                y(:,:,:,3) = y(:,:,:,3) + dummy_y;
            otherwise
                notImpErr
        end
    end
end