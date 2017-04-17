function y = compute_renyi_ef(X, Psi, p, alpha)

    coeff = Psi(X);
    coeff_seq = coeff(:);
    x_abs = abs(coeff_seq);

    x_abs_p = x_abs.^p;
    x_abs_p_sum=sum(x_abs_p);
    x_abs_p_norm = x_abs_p/x_abs_p_sum;
    x_abs_p_norm_alpha = x_abs_p_norm.^alpha;
    y = 1/(1-alpha)*log(sum(x_abs_p_norm_alpha));

end

