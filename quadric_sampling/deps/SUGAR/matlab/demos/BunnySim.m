%Bunny Demo

clear all
close all
load('bunny.mat')
%subset the bunny
N = length(bunny);
N_S=1000;
subset=linspace(0,1,N_S).^2;
subset_r=floor(subset*(N-1))+1;
subset_r=unique(subset_r);
N_S=length(subset_r);
sub_bunny=bunny(subset_r,:);


[sugar_gen_points,~, d_hat, ~, ~, ~, ~, random_points] =sugar(sub_bunny,'equalize',1,'degree_a', 20);
figure

scatter3(sub_bunny(:,3),sub_bunny(:,1),sub_bunny(:,2),20,d_hat,'filled');
title('degree estimate');
colorbar
Zf=[sub_bunny;sugar_gen_points]';
Classes=logical([zeros(1,size(sub_bunny,1)),ones(1,size(sugar_gen_points,1))]);
[~,f1, xvalues1] = DistributionEstimate(sub_bunny,1);
[~,f2, xvalues2, unf] = DistributionEstimate(sugar_gen_points,1);
[~,f3, xvalues3, unf] = DistributionEstimate(random_points,1);

%%
ax1 = subplot(4,4,[1 2 5 6]);
rpts = scatter3(random_points(:,3),random_points(:,1),random_points(:,2), 10,'filled', 'sy', 'MarkerEdgeColor', [1 1 0], 'MarkerEdgeAlpha',1,'MarkerFaceAlpha', 0.5);
hold on
opts = scatter3(Zf(3,~Classes), Zf(1,~Classes),Zf(2,~Classes),10,'filled','p',...
'MarkerFaceAlpha',0.5,'MarkerFaceColor',[0.5 0 0.5], 'MarkerEdgeColor', [0 0 0], 'MarkerEdgeAlpha',1);
view([0 0])
ax2 = subplot(4,4,[3 4 7 8]);
newpts = scatter3(Zf(3,Classes), Zf(1,Classes),Zf(2,Classes),10,'filled',...
    'MarkerFaceAlpha',0.5, 'MarkerEdgeAlpha',1, 'MarkerEdgeColor', 'b')
hold on
opts = scatter3(Zf(3,~Classes), Zf(1,~Classes),Zf(2,~Classes),10,'filled','p',...
'MarkerFaceAlpha',0.5,'MarkerFaceColor',[0.5 0 0.5], 'MarkerEdgeColor', [0 0 0], 'MarkerEdgeAlpha',1);view([90,0])



ax3 = subplot(4,4, [9 10 11 12 13 14 15 16])
emc = plot(xvalues3, unf, 'LineWidth',1, 'Color', 'k');
hold on
scatter(xvalues1, f1,5,'filled','p',...
'MarkerFaceAlpha',0.1,'MarkerFaceColor',[0.5 0 0.5], 'MarkerEdgeColor',...
    [0.5 0 0.5], 'MarkerEdgeAlpha',0.1)
plot(xvalues1,f1,'LineWidth',0.5, 'Color', [0.5 0 0.5]);

scatter(xvalues2, f2,5,'filled','MarkerFaceAlpha',0.1, 'MarkerEdgeColor',...
    'b', 'MarkerEdgeAlpha',0.1, 'MarkerEdgeColor', 'b')
plot(xvalues2, f2,'LineWidth',0.5,'Color', 'b');

rline = scatter(xvalues3, f3,5,'filled','ys','MarkerEdgeColor', [1 1 0], 'MarkerEdgeAlpha',0.1,'MarkerFaceAlpha', 0.1);
plot(xvalues3, f3,'LineWidth',0.5,'Color', 'y');
title("CDF")
hold off

hL = legend([emc,opts,rpts, newpts], {'Empirical CDF', 'Original Points','Random Points', 'New Points'},'Orientation','horizontal')
newPosition = [0.4 0 0.2 0.05];
newUnits = 'normalized';
set(hL,'Position', newPosition,'Units', newUnits);

