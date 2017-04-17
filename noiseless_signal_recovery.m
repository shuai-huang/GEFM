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
relError = zeros(1,4);            % the relative error

% constant, i.e. L1 minimization
Par.p = 1;                        % l1 norm 
Par.X0 = zeros(N, 1);          % initialization
for (lambda_idx=1:200)
    lambda=lambda0*0.9^(lambda_idx-1);
    Xr = ssr_l1(Y, A, Par, lambda);

    if (norm(Xr-Par.X0, 'fro')/norm(Xr, 'fro') < Par.tol) && (lambda<lambda_min)
        break;
    end 
    Par.X0 = Xr; 
end

% for best performance, use the solution from L1 minimization to initialize
Xr_l1 = Xr;
relError(1) = norm(X - Xr_l1, 'fro')/norm(X, 'fro');




% Lp minimization
% initialize with Xr_l1
Par.X0 = Xr_l1;
Par.p = 0.5;    % choose a value around 0.5, needs to be tuned
for (lambda_idx=1:200)
    lambda=lambda0*0.9^(lambda_idx-1);
    Xr = ssr_lp(Y, A, Par, lambda);

    if (norm(Xr-Par.X0, 'fro')/norm(Xr, 'fro') < Par.tol) && (lambda<lambda_min)
        break;
    end 
    Par.X0 = Xr; 
end

Xr_lp = Xr;
relError(2) = norm(X - Xr_lp, 'fro')/norm(X, 'fro');

% Shannon entropy function minimization
% initialize with Xr_l1
Par.X0 = Xr_l1;
Par.p = 1.1;    % choose a value around 1, needs to be tuned
for (lambda_idx=1:200)
    lambda=lambda0*0.9^(lambda_idx-1);
    Xr = ssr_shannon_ef(Y, A, Par, lambda);

    if (norm(Xr-Par.X0, 'fro')/norm(Xr, 'fro') < Par.tol) && (lambda<lambda_min)
        break;
    end 
    Par.X0 = Xr; 
end

Xr_shannon_ef = Xr;
relError(3) = norm(X - Xr_shannon_ef, 'fro')/norm(X, 'fro');


% Renyi entropy function minimization
% initialize with Xr_l1
Par.X0 = Xr_l1;
Par.p = 1.1;        % choose a value around 1, needs to be tuned
Par.alpha = 1.1;    % choose a value around 1, needs to be tuned

for (lambda_idx=1:200)
    lambda=lambda0*0.9^(lambda_idx-1);
    Xr = ssr_renyi_ef(Y, A, Par, lambda);

    if (norm(Xr-Par.X0, 'fro')/norm(Xr, 'fro') < Par.tol) && (lambda<lambda_min)
        break;
    end 
    Par.X0 = Xr; 
end

Xr_renyi_ef = Xr;
relError(4) = norm(X - Xr_renyi_ef, 'fro')/norm(X, 'fro');


relError_mat=[relError_mat; relError];
end




fprintf('Success rate:\n')
fprintf('L1\t:\t %d/10\n', sum(relError_mat(:, 1)<1e-3))
fprintf('Lp\t:\t %d/10\n', sum(relError_mat(:, 2)<1e-3))
fprintf('SEF\t:\t %d/10\n', sum(relError_mat(:, 3)<1e-3))
fprintf('REF\t:\t %d/10\n', sum(relError_mat(:, 4)<1e-3))




