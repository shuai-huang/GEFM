function Xk = srr_shannon_ef_sep( Y, A1, A2, par, lambda, mu)

	% Sparse signal recovery via generalized Shannon entropy function minimization based on FISTA
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

	par.fun = 'shannon_ef';

	fun_sg = str2func( [par.fun '_sg'] );
	fun_compute = str2func( ['compute_' par.fun] );

	kappa              = par.kappa;             % Lipschitz constant
	tol              = par.tol;             % The convergence tolerance
	maxiter       	 = par.maxiter;         % Maximum number of iterations in the main loop
	innermaxiter     = par.innermaxiter;    % Maximum number of iterations in the inner loop
	                                        % It can be set to a small number, 1 usually suffices
	p            	 = par.p;               % The parameter p, it needs to be tuned
	                                        % Usually a number around 1 gives best performance
	Xk               = par.X0;              % Initialize X
	epsilon          = par.epsilon;         % A small positive number, usually set 1e-12


	t_k = 1;
	t_km1 = 1;

	Xkm1 = Xk;

    col_A1 = size(A1,2);
    col_A2 = size(A2,2); 
    
    G = [A1 A2]'*[A1 A2];
    C = [A1 A2]'*Y;   

	for iter = 1 : maxiter
	
		%fprintf('   Inner iteration %d\n', iter);
		Yk = Xk + ((t_km1-1)/t_k)*(Xk-Xkm1);
		Gk = Yk - (1/kappa)*2*(G*Yk-C);

		Xkp1 = [];
		Xk_r = Yk;
        Xkp1_r = Xk_r;
        fpre = kappa/2*norm(Xkp1_r-Gk, 'fro')^2+lambda*fun_compute(Xkp1_r(1:col_A1), p)+lambda*mu*fun_compute(Xkp1_r(col_A1+1:end), p);
		for inneriter  = 1:innermaxiter
			w_1 = fun_sg(Xk_r(1:col_A1), p, epsilon);
			Xkp1_r_1 = weighted_shrinkage(Gk(1:col_A1), lambda/kappa, w_1);

            w_2 = fun_sg(Xk_r(col_A1+1:end), p, epsilon);
            Xkp1_r_2 = weighted_shrinkage(Gk(col_A1+1:end), lambda*mu/kappa, w_2);

            Xkp1_r = [Xkp1_r_1; Xkp1_r_2];

            fcur =  kappa/2*norm(Xkp1_r-Gk, 'fro')^2+lambda*fun_compute(Xkp1_r(1:col_A1), p)+lambda*mu*fun_compute(Xkp1_r(col_A1+1:end), p);
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

% Compute the objective: generalized Shannon entropy function
function y = compute_shannon_ef(X, p)

	x_abs = abs(X);

	x_abs_p = x_abs.^p;
	
	x_abs_p_sum=sum(x_abs_p);
	x_abs_p_norm = x_abs_p/x_abs_p_sum;
	x_abs_p_norm(x_abs_p_norm==0)=1;
	y=-sum(x_abs_p_norm.*log(x_abs_p_norm));

end

% Compute the weights, i.e. first order derivative w.r.t. |x_i|
function w = shannon_ef_sg( x, p, epsilon)

	w = []; 
	
	%computeWeightes
	x_abs = abs(x);
	x_abs_p = x_abs.^p;
	x_abs_p_sum = sum(x_abs_p);

	x_abs_p_ori=x_abs_p;
	x_abs_p(x_abs_p_ori==0)=1;
	x_abs_p_log=log(x_abs_p);
	x_abs_p(x_abs_p_ori==0)=0;

	if ((max(x_abs_p)+epsilon)<x_abs_p_sum)
		w = -log(x_abs_p+epsilon)/x_abs_p_sum + sum(x_abs_p.*x_abs_p_log)/(x_abs_p_sum^2);
		w = p*(x_abs.^(p-1)).*w;
	else
		w = repmat(1, size(x,1), size(x,2));
	end

end
