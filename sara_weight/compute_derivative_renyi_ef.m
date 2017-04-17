function W=compute_derivative_renyi_ef(X, Psi, reg_fun, p, alpha, precision)

W = Psi(X);
W_seq = W(:);

fun_sg = str2func([reg_fun '_sg']);
W = fun_sg(W_seq, p, alpha, precision);
