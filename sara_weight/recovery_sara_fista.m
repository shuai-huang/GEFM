function X_out=recovery_sara_fista(Obs, A, At, lambda, Psi, Psit, Q, par)

    % Image recovery from linear measurements via sparsity averaging: 
    %
    % R. E. Carrillo, et al. “Sparsity averaging for compressive imaging,” IEEE Signal Processing Letters, vol. 20, no. 6, pp. 591–594, June 2013.
    % 
    % By Shuai Huang, The Johns Hopkins University
    % Email: shuang40@jhu.edu 
    % Date: 09/16/2016

    % Obs       : observation/measurement
    % A         : structual random matrix operator
    % At        : transpose of A
    % lambda    : the regularization parameter
    % Psi       : overcomplete wavelet basis operator
    % Psit      : transpose of Psi
    % Q         : a term used by Alternating Split Bregman Shrinkage (ASBS) algorithm
    % par       : various parameters
    
    % Assigning parameres according to par

    maxiter=par.maxiter;            % the maximum number of iterations
    pval=par.pval;                  % the parameter p
    epsilon=par.epsilon;            % a small positive number, can be set to 1e-12
    Xk=par.X0;                      % initialization of estimated image
    kappa = par.kappa;              % the Lipschitz constant
    tol = par.tol;                  % convergence criterion, can be set to 1e-6
    reg_fun = par.reg_fun;          % objective function name
    if (strcmp(reg_fun, 'renyi_ef'))
        alpha = par.alpha;          % if renyi entropy function, set alpha
    end


    % Assign parameters for the denoising procedure 
    parin.denoiseiter=par.denoiseiter;      % max iterations of ASBS algorithm
    parin.innertol=par.innertol;            % convergence criterion, can be set 1e-6
    parin.gamma = par.gamma;                % the gamma parameter in ASBS algorithm              


    t_k = 1;
    t_km1 = 1;
    Xkm1 = Xk;

    fun_val_cur=0;
    fun_compute = str2func( ['compute_' reg_fun] );

    for i=1:maxiter
        % store the old value of the iterate and the t-constant
        Yk = Xk + ((t_km1-1)/t_k)*(Xk-Xkm1);
        
        if (strcmp(reg_fun, 'renyi_ef'))
            W = compute_derivative_renyi_ef(Yk, Psi, reg_fun, pval, alpha, epsilon);
        else
            W=compute_derivative(Yk, Psi, reg_fun, pval, epsilon);
        end
        % gradient step
        T=A(Yk)-Obs;
        B=Yk-1/kappa*2*At(T);

        %invoking the denoising procedure 
        Xkp1 = denoise_asbs(B, lambda, Psi, Psit, W, Q, parin);
        
        fun_val_pre = fun_val_cur;
        fun_val_cur_1 = norm(Obs-A(Xkp1),'fro')^2;
        if (strcmp(reg_fun, 'renyi_ef'))
            fun_val_cur_2 = lambda*fun_compute(Xkp1, Psi, pval, alpha);
        else
            fun_val_cur_2 = lambda*fun_compute(Xkp1, Psi, pval);
        end
        fun_val_cur = fun_val_cur_1 + fun_val_cur_2;
        con_val = abs((fun_val_pre-fun_val_cur)/fun_val_cur);
        
        %con_val = norm(Xkp1 - Xk, 'fro') / norm(Xkp1, 'fro');
        
        %updating t, X
	    t_kp1 = 0.5*(1+sqrt(1+4*t_k*t_k)) ;
	    t_km1 = t_k ;
	    t_k = t_kp1 ;
	    Xkm1 = Xk ;
	    Xk = Xkp1 ;

	
	    fprintf('%3d   %5.5f   %5.5f   %5.5f\n', i, fun_val_cur_1, fun_val_cur_2, con_val)
        if (con_val < tol)
        	break;
        end

    end

    X_out=Xkp1;

end
