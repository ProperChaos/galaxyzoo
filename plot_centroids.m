function plot_centroids(centroids)
    figure(1);
    ha = tight_subplot(20, 20, [.01 .01], [.01 .01], [.01 .01]);

    for j = 1:400
        p = reshape(centroids(j, :), sqrt(size(centroids, 2)/3), sqrt(size(centroids, 2)/3), 3);
        n = (p - min(p(:))) ./ (max(p(:)) - min(p(:)));
        
        axes(ha(j));
        image(n);
        axis square;
        axis off;
        drawnow;
    end
end