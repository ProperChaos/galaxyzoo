function net = train_network(image_features, class_distributions)
%     net = patternnet(50);
%     [net, tr] = train(net, image_features', class_distributions');
%     nntraintool;

      net = TreeBagger(50, image_features, class_distributions, 'Method', 'regression');
end