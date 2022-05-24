function [patches, topleftOrigin] = im2patch(im, patchDim)
% [PATCHES, TOPLEFTORIGIN] = IM2PATCH(IM, PATCHDIM)
% extract overlapping square patches of size PATCHDIM from image IM.
% PATCHES is a matrix of size PATCHDIM^2 x num_patches, the patches are arranged columnwise 
% TOPLEFTORIGIN is amatrix of num_patches x 2 of the column and row indices
% of the upperleft corner of each patch.
%
% Gal Mishne, original version of code by Israel Cohen

[imHeight, imWidth, chan] = size(im);
nInHeight = length(1:1:imHeight-patchDim+1);
nInWidth  = length(1:1:imWidth-patchDim+1);
num_patches = nInHeight*nInWidth;

patches = NaN(patchDim^2*chan, num_patches);

yrange = 0 : (imHeight-patchDim);
xrange = 0 : (imWidth-patchDim);
k = 1;
for c = 1:chan
    for m = 1 : patchDim
        for n = 1 : patchDim
            patches(k,  : ) = reshape(im(n+yrange, m+xrange,c), 1, num_patches);
            k = k+1;
        end
    end
end
cols = 1+xrange;
rows = 1+yrange;
[Cols,Rows] = meshgrid(cols,rows);
topleftOrigin = [Cols(:),Rows(:)];

return;

