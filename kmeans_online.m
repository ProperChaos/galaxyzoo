function [means] = kmeans_online(image_path, d, k, patch_width, patches_per_image)
    means = rand(k, d*patch_width*patch_width)*0.1;
    counts = zeros(k, 1);
    
    % Get directory contents
	listing = dir(image_path);
	filter = [listing.isdir];
	filterData = listing(~filter);
    
    tic;
    
    for it = 1:1000
        for img = 1:size(filterData, 1)
            im = get_img(image_path, filterData, img);
            for j = 1:patches_per_image
                x = data_yield(im, patch_width, d);
                delta = sum(bsxfun(@minus, x', means) .^ 2, 2);
                [~, idx] = min(delta);

                counts(idx, 1) = counts(idx, 1) + 1;
                means(idx, :) = means(idx, :) + (1 / counts(idx, 1)) * (x' - means(idx, :));
            end
            
            if mod(img, 20) == 0
                img / size(filterData, 1) * 100
            end
        end

        it
        t = toc

        if t > 60*60*8
            break;
        end
    end
end

function [im] = get_img(path, listing, img)
    % Read file
    image = listing(img);
    filename = strcat(path, '/', image.name);
    im = imread(filename);
end

function [vect] = data_yield(im, patch_width, d)
    rand = randi(424-patch_width, 2, 1);
    patch = im(rand(1):rand(1)+patch_width-1, rand(2):rand(2)+patch_width-1, 1:d);
    vect = reshape(patch, patch_width*patch_width*d, 1);
    vect = double(vect);
    vect = vect - mean(vect);
    
    if sum(vect) ~= 0
        vect = vect / std(vect);
    end
end