function S_out=recovery_sara_log_nrg_fista(Y, A, At, lambda, Psi, Psit, par)

    % Image recovery from linear measurements via sparsity averaging: 
    %
    % R. E. Carrillo, et al. “Sparsity averaging for compressive imaging,” IEEE Signal Processing Letters, vol. 20, no. 6, pp. 591–594, June 2013.
    % S. Setzer, Split Bregman algorithm, Douglas-Rachford splitting and frame shrinkage, pp. 464–476, Springer Berlin Heidelberg, Berlin, Heidelberg, June 2009.
    % 
    % By Shuai Huang, The Johns Hopkins University
    % Email: shuang40@jhu.edu 
    % Date: 12/20/2018

    % Y         : Observation/measurement
    % A         : structual random matrix operator
    % At        : transpose of A
    % lambda    : the regularization parameter
    % Psi       : overcomplete wavelet basis operator
    % Psit      : inverse overcomplete wavelet transform operator
    % Q         : a term used by Alternating Split Bregman Shrinkage (ASBS) algorithm
    % par       : various parameters
    
    % St        : the recovered images
    % X         : the sparse wavelet coefficient
    
    % Assigning parameres according to par

    maxiter = par.maxiter;          % the maximum number of fista iterations
    pval    = par.pval;             % the parameter p
    epsilon = par.epsilon;          % a small positive number, can be set to 1e-12
    St      = par.X0;               % initialization of estimated image
    kappa   = par.kappa;            % the Lipschitz constant
    tol     = par.tol;              % convergence criterion, can be set to 1e-6
    reg_fun = par.reg_fun;          % objective function name
    if (strcmp(reg_fun, 'renyi_ef'))
        alpha = par.alpha;          % if renyi entropy function, set alpha
    end
    cri_type = par.cri_type;        % criterion type: 0 - relative changes in the image between consecutive iterations
                                    %                 1 - relateive changes in the objective function value between consecutive iterations


    k_t = 1;
    k_tm1 = 1;
    
    Stm1 = St;

    fun_val_cur=0;
    fun_compute = str2func( ['compute_' reg_fun] );
    
    con_val = [];

    for i=1:maxiter
        % store the old value of the iterate and the t-constant
        Rt = St + ((k_tm1-1)/k_t)*(St-Stm1);
        
        if (strcmp(reg_fun, 'renyi_ef'))
            W = compute_derivative_renyi_ef(Rt, Psi, reg_fun, pval, alpha, epsilon);
        else
            W = compute_derivative(Rt, Psi, reg_fun, pval, epsilon);
        end

        % gradient step
        F=Rt-1/kappa*2*At(A(Rt)-Y);

        D = Psi(F);
        D = max(abs(D)-lambda/kappa*W, 0).*sign(D);
        U = Psit(D);
        Stp1 = U;
        
        if (cri_type == 1)
            con_val = norm(Stp1 - St, 'fro') / norm(Stp1, 'fro');
            fprintf('%3d   %5.5f\n', i, con_val)
        else 
            fun_val_pre = fun_val_cur;
            fun_val_cur_1 = norm(Y-A(Stp1))^2;
            if (strcmp(reg_fun, 'renyi_ef'))
                fun_val_cur_2 = lambda*fun_compute(Stp1, Psi, pval, alpha);
            else
                fun_val_cur_2 = lambda*fun_compute(Stp1, Psi, pval);
            end
            fun_val_cur = fun_val_cur_1 + fun_val_cur_2;
            con_val = abs((fun_val_pre-fun_val_cur)/fun_val_cur);
            fprintf('%3d   %5.5f   %5.5f   %5.5f\n', i, fun_val_cur_1, fun_val_cur_2, con_val)
        end
	    
        if (con_val < tol)
        	break;
        end
        
        %updating t, X
	    k_tp1 = 0.5*(1+sqrt(1+4*k_t*k_t)) ;
	    
	    k_tm1 = k_t ;
	    k_t = k_tp1 ;
	    
	    Stm1 = St ;
	    St = Stp1 ;

    end

    S_out=Stp1;

end
