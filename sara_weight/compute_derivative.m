function W=compute_derivative(S, Psi, reg_fun, p, precision)

W = Psi(S);
W_seq = W(:);

fun_sg = str2func([reg_fun '_sg']);
W = fun_sg(W_seq, p, precision);

end
