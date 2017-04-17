function w = renyi_ef_sg( x, p, alpha, precision)
%computeWeightes

w = []; 

%computeWeightes
x_abs = abs(x);
x_abs_p = x_abs.^p;
x_abs_p_sum = sum(x_abs_p);
x_abs_p_norm = x_abs_p/x_abs_p_sum;
x_abs_p_norm_alpha = x_abs_p_norm.^alpha;

p_alpha = p*alpha;

w = 1/(1-alpha)*1/sum(x_abs_p_norm_alpha)*p_alpha/(x_abs_p_sum^(1+alpha));
w = w.*((x_abs+precision).^(p_alpha-1)*x_abs_p_sum - (x_abs+precision).^(p-1)*sum(x_abs.^p_alpha));

end

