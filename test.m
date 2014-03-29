n = zeros(100, 1);

for i = 1:100
    [u,s,v] = svds(centroids, i);
    rec = u * s * v';
    
    delta = rec-centroids;
    n(i) = norm(delta, 'fro');
    
    plot(n);
    drawnow;
end

plot(n);