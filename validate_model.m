function rmse = validate_model(net, features, solutions)
    testY = net.predict(features);
    
    delta = solutions(:, 1) - testY;
    delta = delta .^ 2;
    
    rmse = sqrt(mean(delta(:)));
end