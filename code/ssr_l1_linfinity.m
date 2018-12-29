function Xt = ssr_l1_linfinity( Y, A, par, lambda)

	% Sparse signal recovery via \|x\|_p^p minimization (0<p<1) based on FISTA
	%
    % By Shuai Huang, The Johns Hopkins University
    % Email: shuang40@jhu.edu 
    % Date: 11/26/2016
	%
	% Y             : the observation/measurement vector
	% A             : the sensing matrix
	% par           : various parameters
	% lambda        : the regularization parameter lambda
	

	par.fun = 'l1_linfinity';
	fun_sg = str2func( [par.fun '_sg'] );
	fun_compute = str2func( ['compute_' par.fun] );

	kappa            = par.kappa;             % Lipschitz constant
	tol              = par.tol;             % The convergence tolerance
	maxiter       	 = par.maxiter;         % Maximum number of iterations in the main loop
	Xt               = par.X0;              % Initialize X

	k_t = 1;
	k_tm1 = 0;

	Xtm1 = Xt;
	Xtp1 = [];
	
	Zt = Xt;

	G=A'*A;
	C=A'*Y;

	for iter = 1 : maxiter
	
		%fprintf('   Inner iteration %d\n', iter);
		
		% Compute acceleration update
		Ut = Xt + k_tm1/k_t*(Zt-Xt) + (k_tm1-1)/k_t*(Xt-Xtm1);
		Gt_U = Ut - (1/kappa)*2*(G*Ut-C);
		w_u = fun_sg(Ut);
		Ztp1 = weighted_shrinkage(Gt_U, lambda/kappa, w_u);
		
		% compute non-acceleration update
		Gt_X = Xt - (1/kappa)*2*(G*Xt-C);
		w_x = fun_sg(Xt);
		Vtp1 = weighted_shrinkage(Gt_X, lambda/kappa, w_x);
		
		% compare objective function
		f_Ztp1 = norm(Y-A*Ztp1)^2 + lambda*fun_compute(Ztp1);
		f_Vtp1 = norm(Y-A*Vtp1)^2 + lambda*fun_compute(Vtp1);
		
		if (f_Ztp1<=f_Vtp1)
		    Xtp1 = Ztp1;
		else
		    Xtp1 = Vtp1;
		end

        % check convergence criterion
		if (norm(Xtp1 - Xt) / norm(Xtp1) < tol)
			break;
		end
		
		% update k value
		k_tp1 = 0.5*(1+sqrt(1+4*k_t*k_t));

        % update iterations
		k_tm1 = k_t;
		k_t = k_tp1;
		
		Xtm1 = Xt;
		Xt = Xtp1;
		
		Zt = Ztp1;
		
	end

end


% Compute the weighted soft thresholding 
function X = weighted_shrinkage(Z, lambda, w)

	X = max(abs(Z)-lambda*w, 0) .* sign(Z);
end


% Compute the objective l_1/l_infinity
function y = compute_l1_linfinity(X)

	x_abs = abs(X);
	
	x_abs_max = max(x_abs);
	
	y=sum(x_abs)/length(x_abs)/x_abs_max - 1;

end

% Compute the weights the first order derivative 
function w = l1_linfinity_sg(x)
	% supergradient of l_1/l_infinity
	
    w = [];
    
    % computeWeightes
    x_abs = abs(x);
    x_abs_max = max(x_abs);
    
    x_seq_max = zeros(size(x));
    x_seq_max(x_abs==x_abs_max)=1;
    
    w = (x_abs_max-sum(x_abs)*x_seq_max)/length(x_abs)/(x_abs_max*x_abs_max);
    w = w .* sign(x);

end








































