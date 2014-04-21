function normalize_file(file, M, S)
    out = matfile('15x15_redux/test_normalized.mat', 'Writable', true);
    for i = 1:79975
        out.f(i, 1:12000) = bsxfun(@rdivide, file.f(i, 1:12000) - M, S);
    end
end