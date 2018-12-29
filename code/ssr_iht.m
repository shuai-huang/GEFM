function Xt = ssr_iht(Y, A, par, lambda)

	% Iterative hard thresholding
	% Make sure the operator norm of A is normalized so that ||A||_2<=1
	% Shuai Huang

	tol              = par.tol;
	maxiter       	 = par.maxiter;

	Xt                = par.X0; % Initialize X

	G=A'*A;
	C=A'*Y;

	for iter = 1 : maxiter

        Xt_new = Xt + C - G*Xt;

        Xt_new(abs(Xt_new)<lambda) = 0;

		if (norm(Xt_new - Xt) / norm(Xt_new) < tol)
			break;
		end

        Xt = Xt_new;
		
	end

end

