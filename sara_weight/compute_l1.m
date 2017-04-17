function y = compute_l1(X, Psi, p)

    coeff = Psi(X);
    coeff_seq = coeff(:);
    x_abs = abs(coeff_seq);

    y=sum(x_abs);

end

