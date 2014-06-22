function normalize_files(rows_training, rows_test)
    out1 = matfile('rot_inv/TRAINING_norma.mat', 'Writable', true);
    out2 = matfile('rot_inv/TEST_norma.mat', 'Writable', true);

    in1 = matfile('rot_inv/TRAINING.mat');
    in2 = matfile('rot_inv/TEST.mat');

    M = seq_mean(in1, rows_training);
    S = sqrt(online_variance(in1, rows_training)+0.01);

    for i = 1:rows_training
        out1.f(i, 1:12000) = bsxfun(@rdivide, in1.f(i, 1:12000) - M, S);
    end

    for i = 1:rows_test
        out2.f(i, 1:12000) = bsxfun(@rdivide, in2.f(i, 1:12000) - M, S);
    end
end