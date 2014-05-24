function M = seq_mean(matfile, rows)
    M = zeros(1, 12000);
    for i = 1:rows
        M = M + matfile.f(i, :);
        
        if mod(i, 1000) == 0
            i
        end
    end
    
    M = M ./ rows;
end