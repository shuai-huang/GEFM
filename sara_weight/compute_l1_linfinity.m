function y = compute_l1_linfinity(X, Psi, p)

    coeff = Psi(X);
    coeff_seq = coeff(:);
    x_abs = abs(coeff_seq);

    x_abs_max = max(x_abs);
    y = sum(x_abs)/length(x_abs)/x_abs_max - 1;

end

