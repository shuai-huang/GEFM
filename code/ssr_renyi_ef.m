function Xk = ssr_renyi_ef( Y, A, par, lambda)

	% Sparse signal recovery via generalized renyi entropy function minimization based on FISTA
	%
    % By Shuai Huang, The Johns Hopkins University
    % Email: shuang40@jhu.edu 
    % Date: 11/26/2016
	%
	% Y             : the observation/measurement vector
	% A             : the sensing matrix
	% par           : various parameters
	% lambda        : the regularization parameter lambda
	%

	par.fun = 'renyi_ef';

	fun_sg = str2func( [par.fun '_sg'] );
	fun_compute = str2func( ['compute_' par.fun] );

	kappa              = par.kappa;             % Lipschitz constant
	tol              = par.tol;             % The convergence tolerance
	maxiter       	 = par.maxiter;         % Maximum number of iterations in the main loop
	innermaxiter     = par.innermaxiter;    % Maximum number of iterations in the inner loop
	                                        % It can be set to a small number, 1 usually suffices
	p            	 = par.p;               % The parameter p, it needs to be tuned
	                                        % Usually a number around 1 gives best performance
	alpha			 = par.alpha;           % The parameter alpha, it needs to be tuned
	                                        % Usually a number around 1 gives best performance
	Xk               = par.X0;              % Initialize X
	epsilon          = par.epsilon;         % A small positive number, usually set 1e-12


	t_k = 1;
	t_km1 = 1;

	Xkm1 = Xk;

	G=A'*A;
	C=A'*Y;

	for iter = 1 : maxiter
	
		%fprintf('   Inner iteration %d\n', iter);
		Yk = Xk + ((t_km1-1)/t_k)*(Xk-Xkm1);
		Gk = Yk - (1/kappa)*2*(G*Yk-C);

		Xkp1 = [];
		Xk_r = Yk;
        Xkp1_r = Xk_r;
        fpre = kappa/2*norm(Xkp1_r-Gk, 'fro')^2+lambda*fun_compute(Xkp1_r, p, alpha);
		for inneriter  = 1:innermaxiter
			w = fun_sg(Xk_r, p, alpha, epsilon);
			Xkp1_r = weighted_shrinkage(Gk, lambda/kappa, w);

            fcur = kappa/2*norm(Xkp1_r-Gk, 'fro')^2+lambda*fun_compute(Xkp1_r, p, alpha);
            if (fcur>fpre)
                Xkp1 = Xkp1_r;
                break;
            end
            fpre = fcur;

			if (norm(Xkp1_r - Xk_r, 'fro') / norm(Xkp1_r, 'fro') < tol)
				Xkp1 = Xkp1_r;
				break;
			end
			Xkp1 = Xkp1_r;
			Xk_r = Xkp1_r;
		end

		if (norm(Xkp1 - Xk, 'fro') / norm(Xkp1, 'fro') < tol)
			break;
		end

		t_kp1 = 0.5*(1+sqrt(1+4*t_k*t_k)) ;
		t_km1 = t_k ;
		t_k = t_kp1 ;
		Xkm1 = Xk ;
		Xk = Xkp1 ;
		
	end

end

% Compute the weighted soft thresholding 
function X = weighted_shrinkage(Z, lambda, w)

	X = max(abs(Z)-lambda*w, 0) .* sign(Z);
end

% Compute the objective: generalized Renyi entropy function
function y = compute_renyi_ef(X, p, alpha)

	x_abs = abs(X);
	x_abs_p = x_abs.^p;
	x_abs_p_sum=sum(x_abs_p);
	x_abs_p_norm = x_abs_p/x_abs_p_sum;
	x_abs_p_norm_alpha = x_abs_p_norm.^alpha;
	y = 1/(1-alpha)*log(sum(x_abs_p_norm_alpha));
	
end

% Compute the weights, i.e. first order derivative w.r.t. |x_i|
function w = renyi_ef_sg(x, p, alpha, epsilon)

	w = []; 
	
	%computeWeightes
	x_abs = abs(x);
	x_abs_p = x_abs.^p;
	x_abs_p_sum = sum(x_abs_p);
	x_abs_p_norm = x_abs_p/x_abs_p_sum;
	x_abs_p_norm_alpha = x_abs_p_norm.^alpha;
	
	p_alpha = p*alpha;
	
	w = 1/(1-alpha)*1/sum(x_abs_p_norm_alpha)*p_alpha/(x_abs_p_sum^(1+alpha));
	w = w.*((x_abs+epsilon).^(p_alpha-1)*x_abs_p_sum - (x_abs+epsilon).^(p-1)*sum(x_abs.^p_alpha));

end
