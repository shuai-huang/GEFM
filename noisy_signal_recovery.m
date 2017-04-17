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
rec_snr = zeros(1,4);

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


rec_snr_mat=[rec_snr_mat; rec_snr];
end



fprintf('Mean SNR (dB) of recovered signal:\n')
fprintf('L1  : %.2f dB\n', mean(rec_snr_mat(:,1)))
fprintf('Lp  : %.2f dB\n', mean(rec_snr_mat(:,2)))
fprintf('SEF : %.2f dB\n', mean(rec_snr_mat(:,3)))
fprintf('REF : %.2f dB\n', mean(rec_snr_mat(:,4)))

