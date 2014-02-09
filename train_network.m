function net = train_network(image_features, class_distributions)
    net = patternnet(50);
    [net, tr] = train(net, image_features', class_distributions');
    nntraintool;
end