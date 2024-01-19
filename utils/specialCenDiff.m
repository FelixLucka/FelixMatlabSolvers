function Df = specialCenDiff(f, boundary, dim)
%SPECIALCENDIFF simple spatial central differences for 2D+1D or 3D+1D image sequences
% see "VariationalMethods for Joint Motion Estimation and Image
% Reconstruction" by Hendrik Dirks, 2015, for the details of this approach.
% 
%   Df = specialCenDiff(f,'Hen',1)
%   Df = specialCenDiff(f,'HenAdj',1)

%  INPUTS:
%   f - 3D or 4D numerical array. The last dimension is assumped to
%   correspond to the time dimenions and the first ones to the spatial ones
%   boundary - 'Hen', 'HenAdj', 'Fwd' specifies which boundary conditions
%               are used. 'Hen' and 'HenAdj' correspond to the ones used
%               and described in Hendrik Dirks' dissertation (see above). 
%   dim      - spatial dimension along which to perform the
%              differenciation. 
%
%  OUTPUTS:
%   Df - spatial central differences of f
%
% ABOUT:
%       author          - Felix Lucka
%       date            - ??.??.2017
%       last update     - 27.10.2023
%
% See also


dimSpace = ndims(f)-1;

if(nargin < 3)
    Df = zeros([size(f),dimSpace],'like',f);
    % compute differences in all directions
    switch dimSpace
        case 2
            Df(:,:,:,1) = specialCenDiff(f,boundary,1);
            Df(:,:,:,2) = specialCenDiff(f,boundary,2);
        case 3
            Df(:,:,:,:,1) = specialCenDiff(f,boundary,1);
            Df(:,:,:,:,2) = specialCenDiff(f,boundary,2);
            Df(:,:,:,:,3) = specialCenDiff(f,boundary,3);
    end
    return
else
    Df = zeros(size(f),'like',f);
end


switch dimSpace
    case 2 % dimSpace
        switch dim
            case 1 
                Df(2:end-1,:,:) = (f(3:end,:,:)-f(1:end-2,:,:))/2;
                switch boundary
                    case 'Hen'
                        % nothing to be done, just 0
                    case 'HenAdj'
                        Df(1:2,:,:)       = f(2:3,:,:)/2;
                        Df(end-1:end,:,:) = -f(end-2:end-1,:,:)/2;
                    case 'Fwd'
                        Df([1,end],:,:)   = f([2,end],:,:)-f([1,end-1],:,:);
                    otherwise
                        notImpErr
                end
            case 2
                Df(:,2:end-1,:) = (f(:,3:end,:)-f(:,1:end-2,:))/2;
                switch boundary
                    case 'Hen'
                        % nothing to be done, just 0
                    case 'HenAdj'
                        Df(:,1:2,:)       = f(:,2:3,:)/2;
                        Df(:,end-1:end,:) = -f(:,end-2:end-1,:)/2;
                    case 'Fwd'
                        Df(:,[1,end],:)   = f(:,[2,end],:)-f(:,[1,end-1],:);
                    otherwise
                        notImpErr
                end
            otherwise
                notImpErr
        end
    case 3 % dimSpace
        switch dim
            case 1
                Df(2:end-1,:,:,:) = (f(3:end,:,:,:)-f(1:end-2,:,:,:))/2;
                switch boundary
                    case 'Hen'
                        % nothing to be done, just 0
                    case 'HenAdj'
                        Df(1:2,:,:,:) = f(2:3,:,:,:)/2;
                        Df(end-1:end,:,:,:) = -f(end-2:end-1,:,:,:)/2;
                    case 'Fwd'
                        Df([1,end],:,:,:) = f([2,end],:,:,:)-f([1,end-1],:,:,:);
                    otherwise
                        notImpErr
                end
            case 2
                Df(:,2:end-1,:,:) = (f(:,3:end,:,:)-f(:,1:end-2,:,:))/2;
                switch boundary
                    case 'Hen'
                        % nothing to be done, just 0
                    case 'HenAdj'
                        Df(:,1:2,:,:) = f(:,2:3,:,:)/2;
                        Df(:,end-1:end,:,:) = -f(:,end-2:end-1,:,:)/2;
                    case 'Fwd'
                        Df(:,[1,end],:,:) = f(:,[2,end],:,:)-f(:,[1,end-1],:,:);
                    otherwise
                        notImpErr
                end
            case 3
                Df(:,:,2:end-1,:) = (f(:,:,3:end,:)-f(:,:,1:end-2,:))/2;
                switch boundary
                    case 'Hen'
                        % nothing to be done, just 0
                    case 'HenAdj'
                        Df(:,:,1:2,:) = f(:,:,2:3,:)/2;
                        Df(:,:,end-1:end,:) = -f(:,:,end-2:end-1,:)/2;
                    case 'Fwd'
                        Df(:,:,[1,end],:) = f(:,:,[2,end],:)-f(:,:,[1,end-1],:);
                    otherwise
                        notImpErr
                end
            otherwise
                notImpErr
        end
    otherwise
        notImpErr
end



end