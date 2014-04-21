function variance = online_variance(data)
    n = 0;
    mean = zeros(1, 12000);
    m2 = zeros(1, 12000);
    
    for i = 1:61578
        n = n + 1;
        delta = data.f(i, :) - mean;
        mean = mean + delta / n;
        
        a = data.f(i, :) - mean;
        m2 = m2 + delta.*a;
        
        if mod(i, 100) == 0
            i
        end
    end
    
    variance = m2/(n - 1);
end