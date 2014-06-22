function features = give_features(image_path, centroids, patch_width, stride, P, M)

    % Get directory contents
	listing = dir(strcat(image_path, '\*.jpg'));
	filter = [listing.isdir];
	filterData = listing(~filter);
    
    % amount
    from = 1;
    to = size(filterData, 1);
    
    % Reserve memory
	%features = zeros(to-from+1, 4*size(centroids, 1));
    
    cc = gpuArray(sum(centroids .^ 2, 2)');
    file = matfile('rot_inv/TRAINING.mat', 'Writable', true);
    
    start = tic;

    for j = from:to %1:a
        % For every patch (in total (n-w+1)(n-w+1) patches):
        %   Take patch w-by-w-by-d
        %   Convert to vector N (=w*w*d)
        %   Map to feature vector K
        % Now we have a matrix, K-by-(n-w+1)-by-(n-w+1)
        % Pool in 4 quadrants, gives feature vector of 4K
        tic
        im = imread(strcat(image_path, '\', filterData(j).name));
        
        % Crop to 207x207
        crop_size = 150;
        im = imcrop(im, [(424-crop_size)/2 (424-crop_size)/2 crop_size-1 crop_size-1]);
        
        % Resize to 128x128
        res_size = 15;
        im = imresize(im, [res_size res_size], 'Method', 'bilinear');
        
        % Send to GPU mem
        im = gpuArray(single(im));
        rot = gpuArray(zeros(4, (res_size-patch_width+1)*(res_size-patch_width+1), size(centroids, 1)));
        
        % 4 times!
        for i = 1:4
            im = imrotate(im, 90);
     
            % extract overlapping sub-patches into rows of 'patches'
            % every row is an RGB-channel
            patches = [ im2col(im(:, :, 1), [patch_width patch_width]) ;
                        im2col(im(:, :, 2), [patch_width patch_width]) ;
                        im2col(im(:, :, 3), [patch_width patch_width]) ]';

            % Preprocessing
            patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches, 2)), sqrt(var(patches, [], 2)+10));

            % Whitening
            patches = bsxfun(@minus, patches, M') * P;

            % Activation function
            xx = sum(patches .^ 2, 2);
            xc = 2*(patches * centroids');

            z = sqrt(bsxfun(@plus, cc, bsxfun(@minus, xx, xc)));
            mu = mean(z, 2);
            patches = max(0, bsxfun(@minus, mu, z));
            rot(i, :, :) = patches;
        end

        sums = sum(sum(rot, 3), 2);
        [~, max_idx] = max(sums);
        patches = rot(max_idx, :, :);

        % Reshape
        rows = size(im, 1) - patch_width + 1;
        cols = size(im, 2) - patch_width + 1;
        patches = reshape(patches, rows, cols, size(centroids, 1));

        % Pool
        half_rows = round(rows / 2);
        half_cols = round(cols / 2);

        q1 = sum(sum(patches(1:half_rows, 1:half_cols, :), 1), 2);
        q2 = sum(sum(patches(half_rows+1:end, 1:half_cols, :), 1), 2);
        q3 = sum(sum(patches(1:half_rows, half_cols+1:end, :), 1), 2);
        q4 = sum(sum(patches(half_rows+1:end, half_cols+1:end, :), 1), 2);

        % Concatenate, send to RAM
        file.f(j-from+1, 1:4*size(centroids, 1)) = gather([q1(:);q2(:);q3(:);q4(:)]');

        fprintf('%.2f%% done, total elapsed time was %.2f minutes, elapsed time was %.3f seconds, ETA: %.2f minutes\n', (j-from) / (to-from) * 100.0, toc(start) / 60, toc, toc(start) / (j-from) * (to-j) / 60)
    end
    
    %save('features_training_new_15.mat', 'features');
end