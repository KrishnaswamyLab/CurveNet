function C = plotDiffusionMapinColor(diffusionCoordIm, idxPatches, diffusionMap, inds, figId, plotTitle,step)
% plots the diffusion map with RGB values associated to every sample
% used to correspond between the image and diffusion map = understand what
% coordinates in the diffusion map each pixel was given
%
% Gal Mishne

if ~exist('step','var')
    step = 30;
end
[m, n, k] = size(diffusionCoordIm);
C(:,1) = diffusionCoordIm(idxPatches);
if size(diffusionMap,1) > 1
    C(:,2) = diffusionCoordIm(idxPatches +   m*n);
    C(:,3) = diffusionCoordIm(idxPatches + 2*m*n);
end

if ~isempty(figId)
    figure(figId);
else
    figure;
end
if size(diffusionMap,1) > 5
    subplot(121)
    hh = scatter3(diffusionMap(1,1:step:end),diffusionMap(2,1:step:end),diffusionMap(3,1:step:end),15,C((inds(1:step:end)),:),'filled');
    set(gca,'FontSize',10);
    xlabel('\Psi_1')
    ylabel('\Psi_2')
    zlabel('\Psi_3')
    title(plotTitle);
    axis equal
    subplot(122)
    scatter3(diffusionMap(4,1:step:end),diffusionMap(5,1:step:end),diffusionMap(6,1:step:end),15,C((inds(1:step:end)),:),'filled')
    set(gca,'FontSize',10);
    xlabel('\Psi_4')
    ylabel('\Psi_5')
    zlabel('\Psi_6')
    axis equal
elseif size(diffusionMap,1) >= 3
    scatter3(diffusionMap(1,1:step:end),diffusionMap(2,1:step:end),diffusionMap(3,1:step:end),25,C((inds(1:step:end)),:),'filled')
    set(gca,'FontSize',15);
    xlabel('\Psi_1')
    ylabel('\Psi_2')
    zlabel('\Psi_3')
    title(plotTitle);
elseif size(diffusionMap,1) == 2
    scatter(diffusionMap(1,1:step:end),diffusionMap(2,1:step:end),25,C((inds(1:step:end)),:),'filled')
    set(gca,'FontSize',15);
    xlabel('\Psi_1')
    ylabel('\Psi_2')
    title(plotTitle);
else
    scatter(1:step:size(diffusionMap,2),diffusionMap(1,1:step:end),25,C((inds(1:step:end)),:),'filled')
    set(gca,'FontSize',15);
    xlabel('ind')
    ylabel('\Psi_1')
    title(plotTitle);
end