function features = feature_extractor(images,method,args)
    if method == 'KMEANS'
        features = feature_extractor_kmeans(images, args(1), args(2), args(3), args(4), args(5))
    end
end

function features = feature_extractor_kmeans(images, stride, width, channels, patchCount, nrOfFeatures)
    % Get directory contents
    listing = dir(images);
    
    for image = listing'
        % Process image
        if ~image.isdir
            patches = extract_patches(images, image, width, channels, patchCount);
            
            [~, centroids] = kmeans(patches, nrOfFeatures);
            
            for p = patches'
                feature = zeros(nrOfFeatures, 1);
                for k = 1:nrOfFeatures
                    feature(k) = kmeans_mapping(p, centroids, k);
                end
            end
            
            
        end
    end
end

function f = kmeans_mapping(patch, centroids, k)
    min = 999999;
    min_j = 0;
    for j = 1:size(centroids, 1)
        if norm(centroids(j) - patch) < min
            min = norm(centroids(j) - patch);
            min_j = j;
        end
    end
    
    if min_j == k
        f = 1;
    else
        f = 0;
    end
end

function patches = extract_patches(images, image, width, channels, patchCount)
    filename = strcat(images, '\\', image.name);
    
    % Read image
    im = imread(filename);
    
    patches = zeros(patchCount, channels*width*width);
    
    for i = 1:patchCount
        % Create patch
        rand = randi(424-width, 1);
        rand2 = randi(424-width, 1);
        patch = im(rand:rand+width-1, rand2:rand2+width-1, 1:channels);
        
        % Reshape to R^N
        vect = reshape(patch, width*width*channels, 1);
        
        % Convert to double
        vect = double(vect);
        
        % Normalize
        vect = vect - mean(vect);
        vect = vect / std(vect);
        
        % Whitening
        % TODO
        
        patches(i, :) = vect;
    end
end