function trans_u = opticalFlowOperator(u, v, OF_type, adjoint_fl, para, twist_fl)
%OPTICALFLOWOPERATOR computes the multiplication of the linearized transport operator
% (D_t + v * grad) or (warp_v - Id) or their adjoints with u
%  Du    = opticalFlowOperator(u, v, true, false)
%  Dadju = opticalFlowOperator(u, v, true, true)
%
%  INPUTS:
%   u - 3D or 4D numerical array. The last dimension is assumped to
%   correspond to the time dimenions and the first ones to the spatial ones
%   v - 4D or 5D numerical array of size [size(u), 2] or [size(u), 3] for u
%   being of 3/4 (see above)
%   OF_type     - 'linear' or 'nonLinear' to define which type of optical
%   flow equation should be used
%   adjoint_fl - boolean that indicates whether the forward or the adjoint
%   transport operator should be applied.
%
% OPTIONAL INPUTS
%   para - a struct containing further optional parameters:
%       'warpPara' - parameter for the warping, see warpImage.m
%       'dt' - the time differences between frames as a vector
%             (size T-1 x 1) which will be used to weight the temporal finite
%            differences in the linearized scheme
%   twist_fl - boolean that indicates whether
%            transU_t = warp(u_{t+1},v) - u_{t}  (twist_fl = false, default)
%       or   transU_t = u_{t+1} - warp(u_{t},-v) (twist_fl = true)
%
%  OUTPUTS:
%   transU - result of the application of the optical flow operator or its
%            adjoint with u
%
% ABOUT:
%       author          - Felix Lucka
%       date            - ??.??.????
%       last update     - 27.10.2023
%
% See also

if(nargin < 5)
    para = [];
end

if(nargin < 6)
    twist_fl = false;
end

dt = checkSetInput(para, 'dt', '>0', 1);

sz_u  = size(u);
dim_space = checkSetInput(para, 'dimSpace', [2,3], nDims(u) - 1);
if(length(sz_u) > dim_space)
    T    = sz_u(dim_space+1);
else
    T = 1;
end
sz_u  = sz_u(1:dim_space);

switch dim_space
    case 2
        sqV = @(i) squeeze(v(:,:,i,:));
    case 3
        sqV = @(i) squeeze(v(:,:,:,i,:));
end

switch OF_type
    case 'linear'
        
        if(twist_fl) 
            notImpErr
        end
        
        % apply transU_t = (u_{t+1} - u_{t})/dt + v * grad(u_{t+1})
        
        % forward difference in time
        trans_u = tempFwdDiff(u, adjoint_fl, dt, dim_space);
        
        % central differences in space
        if(any(v(:)))
            switch dim_space
                case 2 % we are in 2D
                    if(~adjoint_fl)
                        trans_u = trans_u + sqV(1) .* specialCenDiff(u(:,:,2:end),'Hen',1) + ...
                                          sqV(2) .* specialCenDiff(u(:,:,2:end),'Hen',2);
                    else
                        trans_u(:,:,2:end) = trans_u(:,:,2:end) - specialCenDiff(sqV(1) .* u,'HenAdj',1);
                        trans_u(:,:,2:end) = trans_u(:,:,2:end) - specialCenDiff(sqV(2) .* u,'HenAdj',2);
                    end
                case 3 % we are in 3D
                    if(~adjoint_fl)
                        trans_u = trans_u + sqV(1) .* specialCenDiff(u(:,:,:,2:end),'Hen',1) + ...
                                          sqV(2) .* specialCenDiff(u(:,:,:,2:end),'Hen',2) + ...
                                          sqV(3) .* specialCenDiff(u(:,:,:,2:end),'Hen',3);
                    else
                        trans_u(:,:,:,2:end) = trans_u(:,:,:,2:end) - specialCenDiff(sqV(1) .* u,'HenAdj',1);
                        trans_u(:,:,:,2:end) = trans_u(:,:,:,2:end) - specialCenDiff(sqV(2) .* u,'HenAdj',2);
                        trans_u(:,:,:,2:end) = trans_u(:,:,:,2:end) - specialCenDiff(sqV(3) .* u,'HenAdj',3);
                    end
                otherwise
                    notImpErr
            end
        end
        
    case 'nonLinear'
        
        warp_para = checkSetInput(para, 'warpPara', 'struct', emptyStruct);
        
        if(~twist_fl)
            % apply transU_t = warp(u_{t+1},v) - u_{t}
            if(~adjoint_fl)
                trans_u = zeros([sz_u, T-1], 'like', u);
                for t = 1:(T-1)
                    switch dim_space
                        case 2
                            trans_u(:,:,t) = warpImage(u(:,:,t+1), v(:,:,:,t), adjoint_fl, warp_para) - u(:,:,t);
                        case 3
                            trans_u(:,:,:,t) = warpImage(u(:,:,:,t+1), v(:,:,:,:,t), adjoint_fl, warp_para) - u(:,:,:,t);
                        otherwise
                            notImpErr
                    end
                end
            else
                T = T + 1;
                trans_u = zeros([sz_u, T], 'like', u);
                for t=1:T-1
                    switch dim_space
                        case 2
                            trans_u(:,:,t)   = trans_u(:,:,t) - u(:,:,t);
                            trans_u(:,:,t+1) = trans_u(:,:,t+1) + warpImage(u(:,:,t), v(:,:,:,t), adjoint_fl, warp_para);
                        case 3
                            trans_u(:,:,:,t)   = trans_u(:,:,:,t) - u(:,:,:,t);
                            trans_u(:,:,:,t+1) = trans_u(:,:,:,t+1) + warpImage(u(:,:,:,t), v(:,:,:,:,t), adjoint_fl, warp_para);
                    end
                    
                end
            end
        else
            % apply transU_t = u_{t+1} - warp(u_{t},-v)
            if(~adjoint_fl)
                trans_u = zeros([sz_u, T-1], 'like', u);
                for t = 1:(T-1)
                    switch dim_space
                        case 2
                            trans_u(:,:,t) = u(:,:,t+1) - warpImage(u(:,:,t), -v(:,:,:,t), adjoint_fl, warp_para);
                        case 3
                            trans_u(:,:,:,t) = - u(:,:,:,t) - warpImage(u(:,:,:,t), -v(:,:,:,:,t), adjoint_fl, warp_para);
                        otherwise
                            notImpErr
                    end
                end
            else
                T = T + 1;
                trans_u = zeros([sz_u, T], 'like', u);
                for t=1:T-1
                    switch dim_space
                        case 2
                            trans_u(:,:,t)   = trans_u(:,:,t) - warpImage(u(:,:,t), -v(:,:,:,t), adjoint_fl, warp_para);
                            trans_u(:,:,t+1) = trans_u(:,:,t+1) + u(:,:,t);
                        case 3
                            trans_u(:,:,:,t)   = trans_u(:,:,:,t) - warpImage(u(:,:,:,t), -v(:,:,:,:,t), adjoint_fl, warp_para);
                            trans_u(:,:,:,t+1) = trans_u(:,:,:,t+1) + u(:,:,:,t);
                    end
                    
                end
            end
        end
end

end