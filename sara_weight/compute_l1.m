function obj_val = compute_l1(S, Psi, p)

    coeff = Psi(S);
    coeff_seq = coeff(:);
    x_abs = abs(coeff_seq);

    obj_val=sum(x_abs);

end

