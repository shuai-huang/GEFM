function W=compute_derivative(X, Psi, reg_fun, p, precision)

W = Psi(X);
W_seq = W(:);

fun_sg = str2func([reg_fun '_sg']);
W = fun_sg(W_seq, p, precision);

end
