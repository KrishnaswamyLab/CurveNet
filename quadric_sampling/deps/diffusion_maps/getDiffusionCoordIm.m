function diffusionCoordIm = getDiffusionCoordIm(im, idxPatches, diffusion_mapX)
% colors the image with RGB values that were assigned to diffusion
% coordinates associated with each pixel
% used to correspond between the image and diffusion map = understand what
% coordinates in the diffusion map each pixel was given
%
% Gal Mishne

[row, col] = ind2sub([size(im,1),size(im,2)],idxPatches);
dimDMcoords = size(diffusion_mapX,1);
if dimDMcoords > 1
    diffusionCoordIm = zeros(size(im,1),size(im,2),3);
    if dimDMcoords == 2
        diffusion_mapX(3,:) = zeros(size(diffusion_mapX(1,:)));
    end
    for i = 1:length(row)
        diffusionCoordIm(row(i),col(i),1) = diffusion_mapX(1,i);
        diffusionCoordIm(row(i),col(i),2) = diffusion_mapX(2,i);
        diffusionCoordIm(row(i),col(i),3) = diffusion_mapX(3,i);
    end
    
    maxval = max(diffusionCoordIm(:));
    minval = min(diffusionCoordIm(:));
    for i = 1:3
        diffusionCoordIm(:,:,i) = (diffusionCoordIm(:,:,i)-minval)/(maxval-minval);
    end
elseif dimDMcoords == 1
    diffusionCoordIm = zeros(size(im,1),size(im,2));
    for i = 1:length(row)
        diffusionCoordIm(row(i),col(i)) = diffusion_mapX(1,i);
    end
    maxval = max(diffusionCoordIm(:));
    minval = min(diffusionCoordIm(:));
    diffusionCoordIm = (diffusionCoordIm-minval)/(maxval-minval);
end

if nargout<1
    gca;imagesc(diffusionCoordIm);
end

return