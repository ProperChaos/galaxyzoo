function rmse = validate_model(net, features, solutions)
    testY = net(features');
    
    delta = solutions' - testY;
    delta = delta .^ 2;
    
    rmse = sqrt(mean(delta(:)));
end