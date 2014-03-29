function features = give_features(image_path, centroids, patch_width, stride)
    % Get directory contents
	listing = dir(image_path);
	filter = [listing.isdir];
	filterData = listing(~filter);
    
    % amount
    from = 43501;
    to = size(filterData, 1);
    
    % Reserve memory
	features = zeros(to-from+1, 4*size(centroids, 1));

    for j = from:to %1:a
        % For every patch (in total (n-w+1)(n-w+1) patches):
        %   Take patch w-by-w-by-d
        %   Convert to vector N (=w*w*d)
        %   Map to feature vector K
        % Now we have a matrix, K-by-(n-w+1)-by-(n-w+1)
        % Pool in 4 quadrants, gives feature vector of 4K
        tic
        im = imread(strcat(image_path, '\', filterData(j).name));
        
        % Crop slightly
        crop_size = 300;
        im = imcrop(im, [(424-crop_size)/2 (424-crop_size)/2 crop_size-1 crop_size-1]);
        
        % Resize
        %im = imresize(im, 0.5);
        
        % Send to GPU mem
        im = single(im);
        
        im_size = size(im);

        % extract overlapping sub-patches into rows of 'patches'
        % every row is an RGB-channel
        patches = [ im2col(im(:, :, 1), [patch_width patch_width]) ;
                    im2col(im(:, :, 2), [patch_width patch_width]) ;
                    im2col(im(:, :, 3), [patch_width patch_width]) ]';
                
        clear im;

        % Preprocessing
        patches = bsxfun(@minus, patches, mean(patches, 2));
        patches = bsxfun(@rdivide, patches, std(patches, 0, 2));

        % Activation function
        xx = sum(patches .^ 2, 2);
        cc = sum(centroids .^ 2, 2)';
        xc = 2*(patches * centroids');
        
        z = sqrt(bsxfun(@plus, cc, bsxfun(@minus, xx, xc)));
        mu = mean(z, 2);
        patches = max(0, bsxfun(@minus, mu, z));
        
        clear xx;
        clear xc;
        clear z;

        % Reshape
        rows = im_size(1) - patch_width + 1;
        cols = im_size(2) - patch_width + 1;
        patches = reshape(patches, rows, cols, size(centroids, 1));

        % Pool
        half_rows = round(rows / 2);
        half_cols = round(cols / 2);

        q1 = sum(sum(patches(1:half_rows, 1:half_cols, :), 1), 2);
        q2 = sum(sum(patches(half_rows+1:end, 1:half_cols, :), 1), 2);
        q3 = sum(sum(patches(1:half_rows, half_cols+1:end, :), 1), 2);
        q4 = sum(sum(patches(half_rows+1:end, half_cols+1:end, :), 1), 2);

        % Concatenate, send to RAM
        features(j-from+1, :) = [q1(:);q2(:);q3(:);q4(:)]';
        j
        toc
        
        if mod(j, 100) == 0
            csvwrite('features_training.csv', features);
        end
    end
    
    % Normalize
    features = bsxfun(@minus, features, mean(features, 2));
    features = bsxfun(@rdivide, features, std(features, 0, 2));
    
    csvwrite('features_training_norma.csv', features)

end