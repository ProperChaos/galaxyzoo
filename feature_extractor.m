function features = feature_extractor(image, centroids, patch_width, stride)
    % For every patch (in total (n-w+1)(n-w+1) patches):
    %   Take patch w-by-w-by-d
    %   Convert to vector N (=w*w*d)
    %   Map to feature vector K
    % Now we have a matrix, K-by-(n-w+1)-by-(n-w+1)
    % Pool in 4 quadrants, gives feature vector of 4K
    
    im = imread(image);
    im = double(im);

    % extract overlapping sub-patches into rows of 'patches'
    % every row is an RGB-channel
    patches = [ im2col(im(:, :, 1), [patch_width patch_width]) ;
                im2col(im(:, :, 2), [patch_width patch_width]) ;
                im2col(im(:, :, 3), [patch_width patch_width]) ]';

    % Preprocessing
    patches = bsxfun(@minus, patches, mean(patches, 2));
    patches = bsxfun(@rdivide, patches, std(patches));

    % Activation function
    xx = sum(patches .^ 2, 2);
    cc = sum(centroids .^ 2, 2)';
    xc = patches * centroids';

    z = sqrt(bsxfun(@plus, cc, bsxfun(@minus, xx, 2*xc)));
    mu = mean(z, 2);
    patches = max(0, bsxfun(@minus, mu, z));

    % Reshape
    rows = size(im, 1) - patch_width + 1;
    cols = size(im, 2) - patch_width + 1;
    patches = reshape(patches, rows, cols, size(centroids, 1));
    
    % Plot all
    figure;
    colormap gray;
    axis image;
    axis off;
    
    c = 16;
    for y = 1:c
        for x = 1:c
            subplot(c, c, (y-1)*c+x);
            a = patches(:, :, (y-1)*c+x);
            b = (a - min(a(:))) / (max(a(:)) - min(a(:)));
            imagesc(b);
            imwrite(b, strcat('C:\\Zoo\\features\\images\\', num2str((y-1)*c+x), '.png'), 'png');
            axis image;
            axis off;
        end
    end

    % Pool
    half_rows = round(rows / 2);
    half_cols = round(cols / 2);
    
    q1 = sum(sum(patches(1:half_rows, 1:half_cols, :), 1), 2);
    q2 = sum(sum(patches(half_rows+1:end, 1:half_cols, :), 1), 2);
    q3 = sum(sum(patches(1:half_rows, half_cols+1:end, :), 1), 2);
    q4 = sum(sum(patches(half_rows+1:end, half_cols+1:end, :), 1), 2);
    
    % Concatenate
    features = [q1(:);q2(:);q3(:);q4(:)]';
end