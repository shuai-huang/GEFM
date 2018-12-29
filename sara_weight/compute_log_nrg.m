function y = compute_log_nrg(S, Psi, p)

    coeff = Psi(S);
    coeff_seq = coeff(:);
    s_abs = abs(coeff_seq);
    s_abs = s_abs(s_abs~=0);

    y=sum(2*log(s_abs));

end

