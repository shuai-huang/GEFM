% add the path, change this if needed
addpath(genpath('./code'))

% problem parameters
N = 1000;           % column number of the sensing matrix
M = 250;            % row number of the sensing matrix
S = 100;            % number of nonzero entries in the signal
std_noise = 0.05;   % standard deviation of the white noise

% function parameters
Par.tol = 1e-6;         % stopping criterion of the FISTA algorithm, make this smaller to improve accuracy
Par.maxiter = 1000;     % the maximum number of iterations of the FISTA algorithm in the main loop
Par.innermaxiter = 1;   % the maximum number of iterations in the inner loop
Par.epsilon = 1e-12;    % add a smalle number to avoid 1/0

rec_snr_mat=[];
for (i=1:100)

fprintf('Recovering %d th sample\n', i)
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

% add white gaussian noise to measurement
noise = randn(M,1) * std_noise;
Y=signal+noise;


% compute the Lipschitz constant
[U, S_diag, V] = eig(A'*A);
Par.kappa = 2*max(diag(S_diag));    % the Lipschitz constant


% start sparse signal recovery
rec_snr = zeros(1,9);

% L1 minimization
Par.X0 = zeros(N, 1);           % initialization
Par.p = 1;	                    % choose a value around 1, needs to be tuned
lambda=0.0725;                  % optimal lambda needs to be tuned

% for best performance, use the solution from L1 minimization to initialize
Xr_l1 = ssr_l1(Y, A, Par, lambda);
rec_snr(1) = snr(X, X-Xr_l1);

% Lp minimization
% initialize with Xr_l1
Par.X0 = Xr_l1;                 % initialization
Par.p = 0.5;                    % choose a value around 0.5, needs to be tuned
lambda=0.0725;                  % optimal lambda needs to be tuned

Xr_lp = ssr_lp(Y, A, Par, lambda);
rec_snr(2) = snr(X, X-Xr_lp);

% Shannon entropy function minimization
% initialize with Xr_l1
Par.X0 = Xr_l1;                 % initialization
Par.p = 1.1;                    % choose a value around 1, needs to be tuned
lambda=5;                       % optimal lambda needs to be tuned

Xr_shannon_ef = ssr_shannon_ef(Y, A, Par, lambda);
rec_snr(3) = snr(X, X-Xr_shannon_ef);

% Renyi entropy function minimization
% initialize with Xr_l1
Par.X0 = Xr_l1;                 % initialization
Par.p = 1.1;                    % choose a value around 1, needs to be tuned
Par.alpha = 1.1;	            % choose a value around 1, needs to be tuned
lambda=5;                       % optimal lambda needs to be tuned

Xr_renyi_ef = ssr_renyi_ef(Y, A, Par, lambda);
rec_snr(4) = snr(X, X-Xr_renyi_ef);


% l_1 / L_infinity function minimization
% initialize with Xr_l1
Par.X0 = Xr_l1;
lambda=1e-6*N;

Xr_l1_linfinity = ssr_l1_linfinity(Y, A, Par, lambda);
rec_snr(5) = snr(X,X-Xr_l1_linfinity);


% Log-NRG via regularized FOCUSS algorithm
% initialize with Xr_l1
Par.X0 = Xr_l1;
Par.maxiter = 1000;
lambda=0.025;
Xr_log_nrg = focuss(Y, A, -1, false, Xr_l1, Par.maxiter, 0, lambda);
rec_snr(6) = snr(X,X-Xr_log_nrg);

% Iterative Hard-thresholding
% make sure the operator norm of A is normalized so that ||A||_2<=1
A_normalized = A/sqrt(max(diag(S_diag)));
Y_normalized = Y/sqrt(max(diag(S_diag)));

Par.X0 = Xr_l1;     % initialization
Par.tol = 1e-6;     % convergence tolerance

lambda = 0.2;

Xr_iht = ssr_iht(Y_normalized, A_normalized, Par, lambda);
rec_snr(7) = snr(X,X-Xr_iht);

% orthogonal matching pursuit
% initialize with Xr_l1
cutoff_thd=0.5;
sparsity_ratio=0.1;    % sparsity upperbound ratio

Xr_l1_init = Xr_l1;
Xr_l1_init(abs(Xr_l1_init)<cutoff_thd)=0;
sparsity0 = length(Xr_l1_init(Xr_l1_init~=0));

sparsity = round(sparsity_ratio*N); % sparsity upperbound

% just to make sure the initialization is sparse, and less than the sparsity upperbound
if (sparsity0>0.8*sparsity)
    Xr_l1_sort = sort(abs(Xr_l1), 'descend');
    Xr_l1_cutoff_thd = Xr_l1_sort(round(0.8*sparsity));

    Xr_l1_init = Xr_l1;
    Xr_l1_init(abs(Xr_l1_init)<Xr_l1_cutoff_thd)=0;
    sparsity0 = length(Xr_l1_init(Xr_l1_init~=0));
end 

omp_opts.X0 = Xr_l1_init;
omp_opts.maxiter = Par.maxiter;
Xr_omp = OMP_init(A, Y, sparsity, omp_opts);
rec_snr(8) = snr(X,X-Xr_omp);

% CoSaMP
% initialize with Xr_l1
cutoff_thd=0.5;
sparsity_ratio=0.06;    % sparsity upperbound ratio

Xr_l1_init = Xr_l1;
Xr_l1_init(abs(Xr_l1_init)<cutoff_thd)=0;
sparsity0 = length(Xr_l1_init(Xr_l1_init~=0));

sparsity = round(sparsity_ratio*N); % sparsity upperbound

% just to make sure the initialization is sparse, and less than the sparsity upperbound
if (sparsity0>0.8*sparsity)
    Xr_l1_sort = sort(abs(Xr_l1), 'descend');
    Xr_l1_cutoff_thd = Xr_l1_sort(round(0.8*sparsity));

    Xr_l1_init = Xr_l1;
    Xr_l1_init(abs(Xr_l1_init)<Xr_l1_cutoff_thd)=0;
    sparsity0 = length(Xr_l1_init(Xr_l1_init~=0));
end 

cosamp_opts.X0 = Xr_l1_init;
cosamp_opts.maxiter = Par.maxiter;
cosamp_opts.normTol = Par.tol;
cosamp_opts.support_tol = Par.tol;

Xr_cosamp = CoSaMP_init_fast(A, Y, sparsity, cosamp_opts);
rec_snr(9) = snr(X,X-Xr_cosamp);


rec_snr_mat=[rec_snr_mat; rec_snr];
end



fprintf('Mean SNR (dB) of recovered signal:\n')
fprintf('L1  : %.2f dB\n', mean(rec_snr_mat(:,1)))
fprintf('Lp  : %.2f dB\n', mean(rec_snr_mat(:,2)))
fprintf('SEF : %.2f dB\n', mean(rec_snr_mat(:,3)))
fprintf('REF : %.2f dB\n', mean(rec_snr_mat(:,4)))
fprintf('L_1-L_infinity : %.2f dB\n', mean(rec_snr_mat(:,5)))
fprintf('Log-NRG : %.2f dB\n', mean(rec_snr_mat(:,6)))
fprintf('IHT : %.2f dB\n', mean(rec_snr_mat(:,7)))
fprintf('OMP : %.2f dB\n', mean(rec_snr_mat(:,8)))
fprintf('CoSaMP : %.2f dB\n', mean(rec_snr_mat(:,9)))

