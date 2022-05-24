close all; clear all;
addpath(genpath(pwd));
rng('shuffle');

quad_coeffs = [0.75, 1];
quad_scales = [16, 32];
is_saddle = [true, false];
Xopt = [0, 0];
Zopt = 0;

rng(19522);
cnt = 1;

for cs = 1:numel(is_saddle)

    for sc = 1:numel(quad_scales)

        a = quad_coeffs(1);
        b = quad_coeffs(2);
        if is_saddle(cs) == true
            b = -b;
        end
    
        a = a * quad_scales(sc);
        b = b * quad_scales(sc);

        Q = @(x) a*(x(1)^2) + b*(x(2)^2);

        % Grid sample for visualization
        [X,Y] = meshgrid(Xopt(1)-1.5:0.005:Xopt(1)+1.5, Xopt(2)-1.5:0.005:Xopt(2)+1.5);
        mesh_pts = [reshape(X, [numel(X), 1]), reshape(Y, [numel(Y), 1])];
        Z = cellfun(Q, num2cell(mesh_pts,2));

        % Sample 500 points around optimum
        sample_pts = Xopt + randsphere(1000, 2, 0.2);
        Z_pts = cellfun(Q, num2cell(sample_pts,2));

        % Sampled points and optimum 
        sample_X = [Xopt(1); sample_pts(:,1)];
        sample_Y = [Xopt(2); sample_pts(:,2)];
        sample_Z = [Zopt; Z_pts];

        % Numerical curvature
        [Xs, Ys] = meshgrid(sample_X, sample_Y);
        f = @(x,y) a.*(x.^2) + b.*(y.^2);
        Zs = f(Xs,Ys);
        [fx,fy] = gradient(Zs);
        [fxx,fxy] = gradient(fx);
        [~,fyy] = gradient(fy);
        K = (fxx.*fyy - fxy.^2)./((1 + fx.^2 + fy.^2).^2);
        H = ((1+fx.^2).*fyy + (1+fy.^2).*fxx - 2.*fx.*fy.*fxy)./...
            ((1 + fx.^2 + fy.^2).^(3/2));

        % params for diffusion
        configParams.t = 8;
        configParams.sigma = 0.5;
        configParams.kNN = 20;
        configParams.num_pts = size(sample_X, 1);

        % pairwise euclidean distance
        euclid_pdist = squareform(pdist([sample_X sample_Y sample_Z]));
        
        % adaptive anisotropic kernel
        sorted_euclid_pdist = sort(euclid_pdist, 2);
        kNN_norm = repmat(sorted_euclid_pdist(:,configParams.kNN), 1, configParams.num_pts);
        kNN_norm_t = kNN_norm';
        %W1 = exp(-euclid_pdist.^2/(configParams.sigma^2));
        W1 = (1/2)*sqrt(2*pi)*(exp(-euclid_pdist.^2./(2*(kNN_norm_t.^2)))./kNN_norm_t + ...
            exp(-euclid_pdist.^2./(2*(kNN_norm.^2)))./kNN_norm);
        D = diag(1./sum(W1,2));
        W = D * W1 * D;

        % diffusion operator
        D = diag(1./sum(W,2));
        P = D * W;
        Pt = P^(configParams.t);

        % diffusion map
        num_eig = configParams.num_pts;
        [v,lambda] = eigs((P + P')/2, num_eig, 'la');
        diff_map_pdist = squareform(pdist(repmat((diag(lambda).^configParams.t)', ...
                           configParams.num_pts, 1) .* v));

        % diffusion curvature
        diff_K = zeros(configParams.num_pts, 1);
        for j = 1:configParams.num_pts
            idx = find(diff_map_pdist(j,:) < prctile(diff_map_pdist(j,:), 5));
            diff_K(j) = mean(Pt(j,idx));
        end

        % select points to display
        plt_disp_idx = find(diff_map_pdist(1,:) < prctile(diff_map_pdist(1,:), 70));
        plt_idx_cmpl = find(diff_map_pdist(1,:) >= prctile(diff_map_pdist(1,:), 70));

        % save points
        save(strcat('toydata_', string(cnt), '.mat'), 'sample_X', 'sample_Y', 'sample_Z');
        
        % visualize surface

        %{
        fig = figure('units','inch','position',[0,0,15,2.5]);

        subplot(1,3,1);
        hold on
        scatter3(sample_pts(:,1), sample_pts(:,2), Z_pts, 5, Z_pts, "red", "filled")
        scatter3(Xopt(1), Xopt(2), Zopt, 15, "white", "filled", "o", "MarkerEdgeColor", "k")
        surf(X, Y, reshape(Z, size(X)), reshape(Z, size(X)));
        shading interp
        xlim([-2.5, 2.5])
        ylim([-2.5, 2.5])
        if b < 0
            zlim([-2, 2])
        else
            zlim([0, 2])
        end
        axis off
        grid off
        colormap(flipud(parula))
        view(3)

        subplot(1,3,2);
        scatter3(sample_X, sample_Y, sample_Z, 3, sample_Z, "filled")
        if b < 0
            zlim([-1, 1])
        else
            zlim([0, 1])
        end
        colormap(flipud(parula))
        colorbar

        subplot(1,3,3);
        contourf(X, Y, reshape(Z, size(X)));
        hold on;
        scatter(sample_pts(:,1), sample_pts(:,2), 1, "red", "filled", "MarkerFaceAlpha", 0.6)
        scatter(Xopt(1), Xopt(2), 3, "white", "filled", "diamond")
        colormap(flipud(parula))
        colorbar

        print(fig, strcat('quadric_', string(cnt)), '-r800', '-dpng');
        %}

        % compute correlation coefficient and mutual information
        Karr = diag(K);
        Ccoeff = round(corrcoef(sort(Karr(plt_disp_idx)),sort(diff_K(plt_disp_idx))),2);
        MI = round(mi(sort(Karr(plt_disp_idx)),sort(diff_K(plt_disp_idx))),2);

        % plot curvature
        fig = figure('units','inch','position',[0,0,15,2.75]);

        ax(1) = subplot(1,3,1);
        scatter3(sample_X(plt_disp_idx), sample_Y(plt_disp_idx), sample_Z(plt_disp_idx), ...
                    3, diff_K(plt_disp_idx), "filled")
        hold on
        scatter3(sample_X(plt_idx_cmpl), sample_Y(plt_idx_cmpl), sample_Z(plt_idx_cmpl), ...
                    3, repmat([0.7, 0.7, 0.7], numel(plt_idx_cmpl), 1), "filled")
        if b < 0
            zlim([-1, 1])
        else
            zlim([0, 1])
        end
        colormap(ax(1),jet)
        colorbar
        hold off

        ax(2) = subplot(1,3,2);
        scatter3(sample_X(plt_disp_idx), sample_Y(plt_disp_idx), sample_Z(plt_disp_idx), ...
                    3, Karr(plt_disp_idx), "filled")
        hold on
        scatter3(sample_X(plt_idx_cmpl), sample_Y(plt_idx_cmpl), sample_Z(plt_idx_cmpl), ...
                    3, repmat([0.7, 0.7, 0.7], numel(plt_idx_cmpl), 1), "filled")
        if b < 0
            zlim([-1, 1])
        else
            zlim([0, 1])
        end
        colormap(ax(2),jet)
        %set(ax(2),'ColorScale','log')
        colorbar
        hold off

        biaxial_plot = [Karr(plt_disp_idx) diff_K(plt_disp_idx)];
        biaxial_plot_ordered = sortrows(biaxial_plot, 1);
        biaxial_plot_smt = smoothdata(biaxial_plot_ordered, 1);

        ax(3) = subplot(1,3,3);
        
        %scatter(Karr(plt_disp_idx), diff_K(plt_disp_idx), 3, "filled")
        %xlabel("Gaussian Curvature")
        %ylabel("Diffusion Curvature")
        annotstr = strcat('Corr. Coeff. :', {' '}, string(Ccoeff(1,2)) , ', MI :', {' '}, string(MI));
        title(annotstr, 'FontSize', 13)
        %xlim([min(biaxial_plot(:,1)), max(biaxial_plot(:,1))])
        %ylim([min(biaxial_plot(:,2)), max(biaxial_plot(:,2))])
        %zoom(0.7)

        yyaxis left
        plot(rescale(1:numel(plt_disp_idx)), sort(Karr(plt_disp_idx)))
        ylabel('Gaussian Curvature')
        hold on
        yyaxis right
        plot(rescale(1:numel(plt_disp_idx)), log(sort(diff_K(plt_disp_idx))))
        ylabel('log(Diffusion Curvature)')
        hold off

        print(fig, strcat('quad_curvature_', string(cnt)), '-r800', '-dpng');

        cnt = cnt + 1;
    
    end

end