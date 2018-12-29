function Xt = ssr_iht_sep(Y, A1, A2, par, lambda, mu)

	% Iterative hard thresholding
	% Make sure the operator norm of A is normalized so that ||A||_2<=1
	% Shuai Huang

	tol              = par.tol;
	maxiter       	 = par.maxiter;

	Xt                = par.X0; % Initialize X
	
	A = [A1 A2];

	G=A'*A;
	C=A'*Y;

	for iter = 1 : maxiter

        Xt_new = Xt + C - G*Xt;
        Xt_new_tr = Xt_new(1:size(A1, 2));
        Xt_new_noise = Xt_new((size(A1, 2)+1) : end);

        Xt_new_tr(abs(Xt_new_tr)<lambda) = 0;
        Xt_new_noise(abs(Xt_new_noise)<lambda*mu) = 0;
        
        Xt_new = [Xt_new_tr; Xt_new_noise];

		if (norm(Xt_new - Xt) / norm(Xt_new) < tol)
			break;
		end

        Xt = Xt_new;
		
	end

end

