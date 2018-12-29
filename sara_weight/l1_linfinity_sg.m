function w = l1_linfinity_sg(x,p,precision)
% supergradient of L_1/L_infinitypenalty

    w = [];
    
    % computeWeightes
    x_abs = abs(x);
    x_abs_max = max(x_abs);
    
    x_seq_max = zeros(size(x));
    x_seq_max(x_abs==x_abs_max)=1;
    
    w = (x_abs_max-sum(x_abs)*x_seq_max)/length(x_abs)/(x_abs_max*x_abs_max);
    w = w .* sign(x);
    
end
