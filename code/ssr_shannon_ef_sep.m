function Xt = srr_shannon_ef_sep( Y, A1, A2, par, lambda, mu)

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
	Xt               = par.X0;              % Initialize X
	epsilon          = par.epsilon;         % A small positive number, usually set 1e-12


	k_t = 1;
	k_tm1 = 0;

	Xtm1 = Xt;
	Xtp1 = [];
	
	Zt = Xt;

    col_A1 = size(A1,2);
    col_A2 = size(A2,2); 
    
    G = [A1 A2]'*[A1 A2];
    C = [A1 A2]'*Y;   


	for iter = 1 : maxiter
	
		%fprintf('   Inner iteration %d\n', iter);
		
		% Compute acceleration update
		Ut = Xt + k_tm1/k_t*(Zt-Xt) + (k_tm1-1)/k_t*(Xt-Xtm1);
		Gt_U = Ut - (1/kappa)*2*(G*Ut-C);
				
		w_u_1 = fun_sg(Ut(1:col_A1), p, epsilon);
		Ztp1_1 = weighted_shrinkage(Gt_U(1:col_A1), lambda/kappa, w_u_1);

        w_u_2 = fun_sg(Ut(col_A1+1:end), p, epsilon);
        Ztp1_2 = weighted_shrinkage(Gt_U(col_A1+1:end), lambda*mu/kappa, w_u_2);
        
		Ztp1 = [Ztp1_1; Ztp1_2];
		
		% compute non-acceleration update
		Gt_X = Xt - (1/kappa)*2*(G*Xt-C);
		
		w_x_1 = fun_sg(Xt(1:col_A1), p, epsilon);
		Vtp1_1 = weighted_shrinkage(Gt_X(1:col_A1), lambda/kappa, w_x_1);

        w_x_2 = fun_sg(Xt(col_A1+1:end), p, epsilon);
        Vtp1_2 = weighted_shrinkage(Gt_X(col_A1+1:end), lambda*mu/kappa, w_x_2);
        
        Vtp1 = [Vtp1_1; Vtp1_2];
        
        % compare objective functions
        f_Ztp1 = norm(Y-[A1 A2]*Ztp1)^2 + lambda*fun_compute(Ztp1_1, p) + lambda*mu*fun_compute(Ztp1_2, p);
        f_Vtp1 = norm(Y-[A1 A2]*Vtp1)^2 + lambda*fun_compute(Vtp1_1, p) + lambda*mu*fun_compute(Vtp1_2, p);
        
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
