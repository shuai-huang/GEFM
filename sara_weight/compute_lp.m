function y = compute_lp(X, Psi, p)

    coeff = Psi(X);
    coeff_seq = coeff(:);
    x_abs = abs(coeff_seq);

    y=sum(x_abs.^p);

end

