function Du = spatialFwdGrad(u, adjoint_fl)
%SPATIALFWDGRAD computes spatial forward gradient for 2D+1D or 3D+1D image sequences
%
%   Du = spatialFwdGrad(u, false)
%   u  = spatialFwdGrad(Du,true)

%  INPUTS:
%   u - 3D or 4D numerical array as [nX nY nT] for 2D+1D and [nX, nY, nZ, nT]
%   for 3D+1D. The last dimension is assumped to correspond to the time dimenions
%   and the first ones to the spatial ones.
%   adjointFL - boolean that idicates whether the forward or the adjoint
%               transport operator should be applied.
%
%   IF THE ADJOINT OPERATOR IS USED, THE SPATIAL INPUT AND OUTPUT
%   DIMENSIONS SWITCH!!!!
%
%
%  OUTPUTS:
%   Du - spatial forward gradient of u as [nX, nY, d, T] for 2D+1D and
%        [nX, nY, nZ, d, nT] for 3D+1D
%
% ABOUT:
%       author          - Felix Lucka
%       date            - ??.??.2017
%       last update     - 27.10.2023
%
% See also

if(~adjoint_fl)
    sz_u      = size(u);
    dim_space = length(sz_u) - 1;
    n_t       = sz_u(end);
    n_xyz     = sz_u(1:end-1);
    Du        = zeros([n_xyz, dim_space, n_t], 'like',u);
    switch dim_space
        case 2 % we are in 2D
            Du(1:end-1,:,1,:) = diff(u,1,1);
            Du(:,1:end-1,2,:) = diff(u,1,2);
        case 3 % we are in 3D
            Du(1:end-1,:,:,1,:) = diff(u,1,1);
            Du(:,1:end-1,:,2,:) = diff(u,1,2);
            Du(:,:,1:end-1,3,:) = diff(u,1,3);
        otherwise
            notImpErr
    end
else
    sz_u      = size(u);
    dim_space = length(sz_u) - 2;
    n_t       = sz_u(end);
    n_xyz     = sz_u(1:end-2);
    dummy_u   = zeros([n_xyz, n_t], 'like',u);
    switch dim_space
        case 2 % we are in 2D
            % fwDX'*ux
            dummy_u(1,:,:) = -u(1,:,1,:);
            dummy_u(2:end-1,:,:) = -diff(u(1:end-1,:,1,:),1,1);
            dummy_u(end,:,:) = u(end-1,:,1,:);
            Du = dummy_u;
            % fwDY'*uy
            dummy_u(:,1,:) = -u(:,1,2,:);
            dummy_u(:,2:end-1,:) = -diff(u(:,1:end-1,2,:),1,2);
            dummy_u(:,end,:) = u(:,end-1,2,:);
            Du = Du + dummy_u;
        case 3 % we are in 2D
            % fwDX'*ux
            dummy_u(1,:,:,:) = -u(1,:,:,1,:);
            dummy_u(2:end-1,:,:,:) = -diff(u(1:end-1,:,:,1,:),1,1);
            dummy_u(end,:,:,:) = u(end-1,:,:,1,:);
            Du = dummy_u;
            % fwDY'*uy
            dummy_u(:,1,:,:) = -u(:,1,:,2,:);
            dummy_u(:,2:end-1,:,:) = -diff(u(:,1:end-1,:,2,:),1,2);
            dummy_u(:,end,:,:) = u(:,end-1,:,2,:);
            Du = Du + dummy_u;
            % fwDZ'*uz
            dummy_u(:,:,1,:) = -u(:,:,1,3,:);
            dummy_u(:,:,2:end-1,:) = -diff(u(:,:,1:end-1,3,:),1,3);
            dummy_u(:,:,end,:) = u(:,:,end-1,3,:);
            Du = Du + dummy_u;
        otherwise
            notImpErr
    end
end