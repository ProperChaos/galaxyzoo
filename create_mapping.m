function centroids = create_mapping(image_path, method, patches_per_image, patch_width, channels, amount_of_centroids)
	if method == 'KMEANS'
		patches = extract_random_patches(image_path, patches_per_image, patch_width, channels);
		centroids = create_mapping_kmeans(patches, amount_of_centroids);
	end
end

function patches = extract_random_patches(image_path, patches_per_image, patch_width, channels)
	% Get directory contents
	listing = dir(image_path);
	filter = [listing.isdir];
	filterData = listing(~filter);
	
	% Reserve memory
	patches = zeros(size(filterData, 1)*patches_per_image, channels * patch_width * patch_width);
	
	j = 1;
	for image = filterData'
		% Read file
		filename = strcat(image_path, '\\', image.name);
		im = imread(filename);
		
		for i = 1:patches_per_image
			% Get random patch
			rand = randi(424-patch_width, 2, 1);
			patch = im(rand(1):rand(1)+patch_width-1, rand(2):rand(2)+patch_width-1, 1:channels);
			
			% Reshape to R^N
			vect = reshape(patch, patch_width*patch_width*channels, 1);
			
			% Convert to double
			vect = double(vect);
			
			% Normalize local brightness and contrast
			vect = vect - mean(vect);
			vect = vect / std(vect);
			
			% Whitening
			% TODO
			
			% Set patch
			patches((j-1)*patches_per_image + i, :) = vect;
		end
		
		j = j + 1;
        
        if mod(j, 1000) == 0
            j/size(filterData, 1)
        end
	end
end

function centroids = create_mapping_kmeans(patches, amount_of_centroids)
	[~, centroids] = kmeans(patches, amount_of_centroids);
end