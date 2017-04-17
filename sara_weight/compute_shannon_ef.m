function y = compute_shannon_ef(X, Psi, p)

    coeff = Psi(X);
    coeff_seq = coeff(:);
    x_abs = abs(coeff_seq);


    x_abs = x_abs.^p;

    x_abs_sum=sum(x_abs);
    x_abs_norm = x_abs/x_abs_sum;
    x_abs_norm(x_abs_norm==0)=1;
    y=-sum(x_abs_norm.*log(x_abs_norm));

end

