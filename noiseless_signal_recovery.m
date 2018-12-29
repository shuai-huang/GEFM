% add the path, change this if needed
addpath(genpath('./code'))

% problem parameters
N = 1000;    % column number of the sensing matrix
M = 500;     % row number of the sensing matrix
S = 275;     % number of nonzero entries in the signal

% function parameters
Par.tol = 1e-6;         % stopping criterion of the FISTA algorithm, make this smaller to improve accuracy
Par.maxiter = 1000;     % the maximum number of iterations of the FISTA algorithm in the main loop
Par.innermaxiter = 1;   % the maximum number of iterations in the inner loop
Par.epsilon = 1e-12;    % add a smalle number to avoid 1/0


lambda0 = 0.25;     % For noiseless recovery, start with a relative large lambda0 and solve the recovery problem iteratively with a decreasing lambda sequence
lambda_min = 1e-8;  % The smallest lambda value


relError_mat = [];
for (i=1:10)

fprintf('Recovering %d-th sample\n', i)

% sensing matrix created using a random gaussian distribution N(0,1)
A = randn(M,N);
A_norm=sqrt(sum(A.^2));
for(j=1:N)
    A(:,j)=A(:,j)/A_norm(j);
end 

% Generate s random gaussian sources (i.i.d. sources)
nonzeroW = randn(S, 1);
% select active sources at random locations
ind = randperm(N);
indice = ind(1 : S);
X = zeros(N, 1);
X(indice,:) = nonzeroW;

% noiseless measurement
signal = A * X;
Y = signal;

% compute the Lipschitz constant
[U, S_diag, V] = eig(A'*A);
Par.kappa = 2*max(diag(S_diag));    % the Lipschitz constant


% start sparse signal recovery
relError = zeros(1,9);            % the relative error

% constant, i.e. L1 minimization
Par.p = 1;                        % l1 norm 
Par.X0 = zeros(N, 1);          % initialization
for (lambda_idx=1:200)
    lambda=lambda0*0.9^(lambda_idx-1);
    Xr = ssr_l1(Y, A, Par, lambda);

    if (norm(Xr-Par.X0)/norm(Xr) < Par.tol) && (lambda<lambda_min)
        break;
    end 
    Par.X0 = Xr; 
end

% for best performance, use the solution from L1 minimization to initialize
Xr_l1 = Xr;
relError(1) = norm(X - Xr_l1)/norm(X);


% Lp minimization
% initialize with Xr_l1
Par.X0 = Xr_l1;
Par.p = 0.5;    % choose a value around 0.5, needs to be tuned
for (lambda_idx=1:200)
    lambda=lambda0*0.9^(lambda_idx-1);
    Xr = ssr_lp(Y, A, Par, lambda);

    if (norm(Xr-Par.X0)/norm(Xr) < Par.tol) && (lambda<lambda_min)
        break;
    end 
    Par.X0 = Xr; 
end

Xr_lp = Xr;
relError(2) = norm(X - Xr_lp)/norm(X);

% Shannon entropy function minimization
% initialize with Xr_l1
Par.X0 = Xr_l1;
Par.p = 1.1;    % choose a value around 1, needs to be tuned
for (lambda_idx=1:200)
    lambda=lambda0*0.9^(lambda_idx-1);
    Xr = ssr_shannon_ef(Y, A, Par, lambda);

    if (norm(Xr-Par.X0)/norm(Xr) < Par.tol) && (lambda<lambda_min)
        break;
    end 
    Par.X0 = Xr; 
end

Xr_shannon_ef = Xr;
relError(3) = norm(X - Xr_shannon_ef)/norm(X);


% Renyi entropy function minimization
% initialize with Xr_l1
Par.X0 = Xr_l1;
Par.p = 1.1;        % choose a value around 1, needs to be tuned
Par.alpha = 1.1;    % choose a value around 1, needs to be tuned

for (lambda_idx=1:200)
    lambda=lambda0*0.9^(lambda_idx-1);
    Xr = ssr_renyi_ef(Y, A, Par, lambda);

    if (norm(Xr-Par.X0)/norm(Xr) < Par.tol) && (lambda<lambda_min)
        break;
    end 
    Par.X0 = Xr; 
end

Xr_renyi_ef = Xr;
relError(4) = norm(X - Xr_renyi_ef)/norm(X);


% l_1 / L_infinity function minimization
% initialize with Xr_l1
Par.X0 = Xr_l1;
lambda0_l1_linfinity = lambda0*1e-6*N;

for (lambda_idx=1:200)
    lambda=lambda0_l1_linfinity*0.9^(lambda_idx-1);
    Xr = ssr_l1_linfinity(Y, A, Par, lambda);

    if (norm(Xr-Par.X0)/norm(Xr) < Par.tol) && (lambda<lambda_min)
        break;
    end 
    Par.X0 = Xr; 
end

Xr_l1_linfinity = Xr;
relError(5) = norm(X - Xr_l1_linfinity)/norm(X);


% logarithm of energy minimization using regularized FOCUSS algorithm
% initialize with Xr_l1
Par.X0 = Xr_l1;
Xr_log_nrg = focuss(Y, A, -1, false, Par.X0, Par.maxiter, 0, 0);
relError(6) = norm(X - Xr_log_nrg)/norm(X);


% iterative hard thresholding
% make sure the operator norm of A is normalized so that ||A||_2<=1
A_normalized = A/sqrt(max(diag(S_diag)));
Y_normalized = Y/sqrt(max(diag(S_diag)));

Par.X0 = Xr_l1;

for (lambda_idx=1:200)
    lambda=lambda0*0.9^(lambda_idx-1);
    Xr = ssr_iht(Y_normalized, A_normalized, Par, lambda);

    if (norm(Xr-Par.X0)/norm(Xr) < Par.tol) && (lambda<lambda_min)
        break;
    end
    Par.X0 = Xr;
end

Xr_iht = Xr;
relError(7) = norm(X - Xr_iht)/norm(X);


% orthogonal matching pursuit
% initialize with Xr_l1
Xr_l1_cutoff_thd = lambda0;
Xr_l1_init = Xr_l1;
Xr_l1_init(abs(Xr_l1_init)<Xr_l1_cutoff_thd)=0;

omp_opts.X0 = Xr_l1_init;
omp_opts.maxiter = Par.maxiter;

sparsity = length(Y);   % set an upper bound for sparsity
Xr = OMP_init(A, Y, sparsity, omp_opts);

Xr_omp = Xr;
relError(8) = norm(X - Xr_omp)/norm(X);


% CoSaMP
% initialize with Xr_constant
Xr_l1_cutoff_thd = lambda0;

Xr_l1_init = Xr_l1;
Xr_l1_init(abs(Xr_l1_init)<Xr_l1_cutoff_thd)=0;
sparsity0 = length(Xr_l1_init(Xr_l1_init~=0));  % sparsity of the initialization

cosamp_opts.X0 = Xr_l1_init;
cosamp_opts.maxiter = 200;
cosamp_opts.normTol = Par.tol;
cosamp_opts.support_tol = Par.tol;

% record the residue and select the smallest one among them
residue_seq=[];
Xr_mat=[];
for (sparsity_idx=1:N)
    sparsity = sparsity0 + sparsity_idx;

    if (sparsity>=S+5)  % a liitle trick to avoid running indefinitely, no need to venture further than the oracle sparsity...
        break;
    end

    Xr = CoSaMP_init_fast(A, Y, sparsity, cosamp_opts);
    Xr_mat = [Xr_mat Xr];
    residue_seq = [residue_seq norm(Y-A*Xr)];
end

[residue_min, residue_min_idx]=min(residue_seq);
Xr = Xr_mat(:, residue_min_idx);

Xr_cosamp = Xr;
relError(9) = norm(X - Xr_cosamp)/norm(X);


relError_mat=[relError_mat; relError];
end




fprintf('Success rate:\n')
fprintf('L1\t:\t %d/10\n', sum(relError_mat(:, 1)<1e-3))
fprintf('Lp\t:\t %d/10\n', sum(relError_mat(:, 2)<1e-3))
fprintf('SEF\t:\t %d/10\n', sum(relError_mat(:, 3)<1e-3))
fprintf('REF\t:\t %d/10\n', sum(relError_mat(:, 4)<1e-3))
fprintf('L_1-L_infinity\t:\t %d/10\n', sum(relError_mat(:, 5)<1e-3))
fprintf('log-NRG\t:\t %d/10\n', sum(relError_mat(:, 6)<1e-3))
fprintf('IHT\t:\t %d/10\n', sum(relError_mat(:, 7)<1e-3))
fprintf('OMP\t:\t %d/10\n', sum(relError_mat(:, 8)<1e-3))
fprintf('CoSaMP\t:\t %d/10\n', sum(relError_mat(:, 9)<1e-3))



