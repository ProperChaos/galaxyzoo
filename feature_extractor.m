function features = feature_extractor(image, centroids, patch_width, channels, stride)
    % For every patch (in total (n-w+1)(n-w+1) patches):
    %   Take patch w-by-w-by-d
    %   Convert to vector N (=w*w*d)
    %   Map to feature vector K
    % Now we have a matrix, K-by-(n-w+1)-by-(n-w+1)
    
    % Pool in 4 quadrants, gives feature vector of 4K
    
    im = imread(image);
    
    i = 1;
    j = 1;
    
    patch_features = zeros(size(centroids, 1), (424-patch_width)/stride + 1, (424-patch_width)/stride + 1);
    
    tic
    
    while j < 424-patch_width+1
        while i < 424-patch_width+1
            patch = im(i:i+patch_width-1, j:j+patch_width-1, 1:channels);
            
            % Process patch
            patch_features(:, (i-1)/stride+1, (j-1)/stride+1) = process_patch(patch, patch_width, channels, centroids);
            
            i = i + stride;
        end
        i = 1;
        j = j + stride;
        j
    end
    
    toc
    
    % Pool
    features = sum_pool(patch_features, 4);
end

function proc = process_patch(patch, patch_width, channels, centroids)
    % Reshape to R^N
    vect = reshape(patch, patch_width*patch_width*channels, 1);

    % Convert to double
    vect = double(vect);

    % Normalize local brightness and contrast
    vect = vect - mean(vect);
    vect = vect / std(vect);
    
    proc = map_kmeans(vect, centroids);
end

function feature_vector = map_kmeans(vect, centroids)
    z = bsxfun(@minus, vect, centroids');
    twoNorm = sqrt(sum(abs(z).^2, 1));
    
    mu = mean(twoNorm);

    feature_vector = max(0, bsxfun(@minus, mu, twoNorm));
end