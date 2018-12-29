function y = compute_iht(X, Psi, lambda)

    coeff = Psi(X);
    coeff_seq = coeff(:);
    x_abs = abs(coeff_seq);

    y=sum(x_abs);   % for iht this function value is not the true l0 norm value, t just serves as a probe to the optimization process

end

