function J = energyTVTVOF(u, v, alpha, beta, gamma, para)
%ENERGYTVTVOF computes the energy 
%   \alpha \|\nabla u_t \|_1 
% + \beta \sum_i^d \|\nabla v_{x_i,t} \|_1 
% +  \gamma/p \| u_{t+1} - u_t + (\nabla u_t) \cdot v_t \|_p^p } 
% for a motivation of this, see 
% "Enhancing Compressed Sensing Photoacoustic Tomography by Simultaneous 
% Motion Estimation" by Arridge, Beard, Betcke, Cox, Huynh, Lucka Zhang,
% 2017. 
%
% DESCRIPTION:
%       energyTVTVOF computes the regularization energy used by the joint
%       image reconstruction and optical flow estimation method
%
% USAGE:
%       J = energyTVTVOF(u, v, alpha, beta, gamma, para)
%
% INPUTS:
%       u - 2D+1D or 3D+1D numerical array
%       v - 2D+1D+1D or 3D+1D+1D numerical array
%       alpha - regularization parameter for spatial TV for u
%       beta  - regularization parameter for spatial TV for v
%       gamma - regularization parameter for motion coupling
%       para  - struct containing additional parameters:
%
%         energy function specification:            
%           'OpticalFlowConstraint' - norm to implement optical flow constraint 
%                             'L1' or 'L2' (df)
%           'TVtypeU' - total variation type used for image 
%                       'isotropic' (df) or 'anisotropic'
%           'TVtypeV' - total variation type used for motion field
%                       'anisotropic', 'mixedIsotropic' (df), 'fullyIsotropic'
%           'constraint' - constraint on image: 'none'�(df), 'positivity' 'range'
%
% 
%
% OUTPUTS:
%       J - value of the energy
% 
% 
%
% ABOUT:
%   author          - Felix Lucka based on code by Hendrik Dirks:
%                     https://github.com/HendrikMuenster/JointMotionEstimationAndImageReconstruction
%   date            - 06.05.2018
%   last update     - 21.12.2023
%
% See also 

dim_u        = size(u);
dim_v        = size(v);
dim_space    = length(dim_u(1:end-1));

opt_flw_con_type = checkAndAssignStruct(para,'OpticalFlowConstraint', {'L1','L2'}, 'L2');
constraint       = checkAndAssignStruct(para,'constraint', {'none', 'positivity', 'range'}, 'none');

J = Inf;

% check constraints
switch constraint
    case 'none'
        con_violation_fl = false;
    case 'positivity'
        con_violation_fl = any(u(:) < 0);
    case 'range'
        con_range = checkAndAssignStruct(para,'range','double','error');
        if(~ length(con_range) == 2 && con_range(1) < con_range(2))
            error(['invalid range: ' num2str(con_range)])
        end
        con_violation_fl = any(u(:) < con_range(1)) | any(u(:) > con_range(2));
end

if(con_violation_fl)
    return
end

TV_u              = sumAll(sqrt(sum((spatialFwdGrad(u,false)).^2,length(dim_v))));
optical_flow_part = transportTerm(u,v,false);

switch dim_space
    case 2        
        % forward gradients of v
        d_fwd_x_v = zeros(dim_v,'like',v);
        d_fwd_x_v(1:end-1,:,:,:) = diff(v,1,1);
        d_fwd_y_v = zeros(dim_v,'like',v);
        d_fwd_y_v(:,1:end-1,:,:) = diff(v,1,2);
        TV_v = sum(sqrt(d_fwd_x_v(:).^2+d_fwd_y_v(:).^2));
    case 3
        % forward gradients of v
        d_fwd_x_v = zeros(dim_v,'like',v);
        d_fwd_x_v(1:end-1,:,:,:,:) = diff(v,1,1);
        d_fwd_y_v = zeros(dim_v,'like',v);
        d_fwd_y_v(:,1:end-1,:,:,:) = diff(v,1,2);
        d_fwd_z_v = zeros(dim_v,'like',v);
        d_fwd_z_v(:,:,1:end-1,:,:) = diff(v,1,3);
        TV_v = sum(sqrt(d_fwd_x_v(:).^2+d_fwd_y_v(:).^2+d_fwd_z_v(:).^2));
end

switch opt_flw_con_type
    case 'L2'
        optical_flow = 0.5*sum(optical_flow_part(:).^2);
    case 'L1'
        optical_flow = sum(abs(optical_flow_part(:)));
end

J = alpha * TV_u + beta * TV_v + gamma * optical_flow;


end