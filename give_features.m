function features = give_features(image_path, centroids, patch_width, stride)
    % Get directory contents
	listing = dir(image_path);
	filter = [listing.isdir];
	filterData = listing(~filter);
    
    % Reserve memory
	features = zeros(size(filterData, 1), 4*size(centroids, 1));

	j = 1;
    figure;
	for image = filterData'
        tic;
        
        features(j, :) = feature_extractor(strcat(image_path, '\\', image.name), centroids, patch_width, stride);
        
        times(1, j) = toc;

        avg = mean(times, 2);
        eta = round((size(filterData, 1) - j) * avg / 60);
        disp(strcat('ETA: ', num2str(eta), ' minutes'));
        
        j = j+1;
    end
end