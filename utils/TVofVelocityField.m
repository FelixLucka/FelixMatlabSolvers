function TVv = TVofVelocityField(v, TV_type, single_frame_flag)
%TVOFVELOCITYFIELD computes the total variation of a velocity field
%
% DESCRIPTION:
%       TVofVelocityField computes different ways to define the total
%       variation of a velocity field
%
% USAGE:
%       TVv = TVofVelocityField(v,TVtype,singleFrameFL)
%
% INPUTS:
%       v - 2D+1D+1D or 3D+1D+1D numerical array
%       TVtype - total variation type used for motion field
%                'anisotropic', 'mixedIsotropic', 'fullyIsotropic'
%       single_frame_flag - boolean indicating whether the input field is
%       static, i.e., only a single frame
% 
%
% OUTPUTS:
%       TVv - total variation of v
%
% ABOUT:
%   based on code by Hendrik Dirks,
%   https://github.com/HendrikMuenster/JointMotionEstimationAndImageReconstruction
%
%   author          - Felix Lucka
%   date            - 06.05.2018
%   last update     - 23.12.2018
%
% See also



if(nargin < 3)
    single_frame_flag = false;
end


if(single_frame_flag)
    dim_space = ndims(v)-1;
    Dv        = spatialFwdJacobian(v,false,true);
    switch TV_type
        case 'anisotropic'
            TVv = sumAll(abs(Dv));
        case 'mixedIsotropic'
            TVv = sumAll(sqrt(sum(Dv.^2, dim_space+2)));
        case 'fullyIsotropic'
            TVv = sumAll(sqrt(sum(sum(Dv.^2, dim_space+2),dim_space+1)));
    end
else
    dim_space = ndims(v)-2;
    Dv = spatialFwdJacobian(v,false);
    switch TV_type
        case 'anisotropic'
            TVv = sumAllBut(abs(Dv),dim_space+1);
        case 'mixedIsotropic'
            TVv = sumAllBut(sqrt(sum(Dv.^2,dim_space+3)),dim_space+1);
        case 'fullyIsotropic'
            TVv = sumAllBut(sqrt(sum(sum(Dv.^2,dim_space+3),dim_space+2)),dim_space+1);
    end
end


end