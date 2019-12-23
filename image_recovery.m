%    v1.0 (SH) - First release (03/30/2017)
%    v1.1 (SH) - Second release (01/24/2018)
%    v2.0 (SH) - Third release (12/29/2018)
%    v2.1 (SH) - (12/23/2019) update the image recovery algorithm with faster implementation that does not rely on alternating the split bregman shrinkage algorithm

addpath(genpath('./sara_weight'));

rate = 0.2;             % sampling rate
image_name = 'lena';    % image name
wave_num = 4;           % overcomplete wavelet basis are constructed by concatenating db1-db4, usually this would suffice
load(strcat('./test_images_256/', image_name, '.mat'));
im=zeros(256,256);
nlevel=4;

% construct the sensing matrix
dwtmode('per');
[C,S]=wavedec2(im,nlevel,'db8'); 
ncoef=length(C);
[C1,S1]=wavedec2(im,nlevel,'db1'); 
ncoef1=length(C1);
[C2,S2]=wavedec2(im,nlevel,'db2'); 
ncoef2=length(C2);
[C3,S3]=wavedec2(im,nlevel,'db3'); 
ncoef3=length(C3);
[C4,S4]=wavedec2(im,nlevel,'db4'); 
ncoef4=length(C4);
[C5,S5]=wavedec2(im,nlevel,'db5'); 
ncoef5=length(C5);
[C6,S6]=wavedec2(im,nlevel,'db6'); 
ncoef6=length(C6);
[C7,S7]=wavedec2(im,nlevel,'db7'); 
ncoef7=length(C7);

switch wave_num
case 1
    Psi = @(x) [wavedec2(x,nlevel,'db1')']/sqrt(wave_num); 
    Psit = @(x) (waverec2(x(1:ncoef1),S1,'db1'))/sqrt(wave_num);

case 2
    Psi = @(x) [wavedec2(x,nlevel,'db1')'; wavedec2(x,nlevel,'db2')']/sqrt(wave_num); 
    Psit = @(x) (waverec2(x(1:ncoef1),S1,'db1') + waverec2(x(ncoef1+1:ncoef1+ncoef2),S2,'db2'))/sqrt(wave_num);

case 3
    Psi = @(x) [wavedec2(x,nlevel,'db1')'; wavedec2(x,nlevel,'db2')';wavedec2(x,nlevel,'db3')']/sqrt(wave_num); 
    Psit = @(x) (waverec2(x(1:ncoef1),S1,'db1') + waverec2(x(ncoef1+1:ncoef1+ncoef2),S2,'db2') + waverec2(x(ncoef1+ncoef2+1:ncoef1+ncoef2+ncoef3),S3,'db3'))/sqrt(wave_num);

case 4
    Psi = @(x) [wavedec2(x,nlevel,'db1')'; wavedec2(x,nlevel,'db2')';wavedec2(x,nlevel,'db3')'; wavedec2(x,nlevel,'db4')']/sqrt(wave_num); 
    Psit = @(x) (waverec2(x(1:ncoef1),S1,'db1') + waverec2(x(ncoef1+1:ncoef1+ncoef2),S2,'db2') + waverec2(x(ncoef1+ncoef2+1:ncoef1+ncoef2+ncoef3),S3,'db3') + waverec2(x(ncoef1+ncoef2+ncoef3+1:ncoef1+ncoef2+ncoef3+ncoef4),S4,'db4'))/sqrt(wave_num);

case 5
    Psi = @(x) [wavedec2(x,nlevel,'db1')'; wavedec2(x,nlevel,'db2')';wavedec2(x,nlevel,'db3')'; wavedec2(x,nlevel,'db4')'; wavedec2(x,nlevel,'db5')']/sqrt(wave_num); 
    Psit = @(x) (waverec2(x(1:ncoef1),S1,'db1') + waverec2(x(ncoef1+1:ncoef1+ncoef2),S2,'db2') + waverec2(x(ncoef1+ncoef2+1:ncoef1+ncoef2+ncoef3),S3,'db3') + waverec2(x(ncoef1+ncoef2+ncoef3+1:ncoef1+ncoef2+ncoef3+ncoef4),S4,'db4') + waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5),S5,'db5'))/sqrt(wave_num);

case 6
    Psi = @(x) [wavedec2(x,nlevel,'db1')'; wavedec2(x,nlevel,'db2')';wavedec2(x,nlevel,'db3')'; wavedec2(x,nlevel,'db4')'; wavedec2(x,nlevel,'db5')'; wavedec2(x,nlevel,'db6')']/sqrt(wave_num); 
    Psit = @(x) (waverec2(x(1:ncoef1),S1,'db1') + waverec2(x(ncoef1+1:ncoef1+ncoef2),S2,'db2') + waverec2(x(ncoef1+ncoef2+1:ncoef1+ncoef2+ncoef3),S3,'db3') + waverec2(x(ncoef1+ncoef2+ncoef3+1:ncoef1+ncoef2+ncoef3+ncoef4),S4,'db4') + waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5),S5,'db5') + waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6),S6,'db6'))/sqrt(wave_num);

case 7
    Psi = @(x) [wavedec2(x,nlevel,'db1')'; wavedec2(x,nlevel,'db2')';wavedec2(x,nlevel,'db3')'; wavedec2(x,nlevel,'db4')'; wavedec2(x,nlevel,'db5')'; wavedec2(x,nlevel,'db6')'; wavedec2(x,nlevel,'db7')']/sqrt(wave_num); 
    Psit = @(x) (waverec2(x(1:ncoef1),S1,'db1') + waverec2(x(ncoef1+1:ncoef1+ncoef2),S2,'db2') + waverec2(x(ncoef1+ncoef2+1:ncoef1+ncoef2+ncoef3),S3,'db3') + waverec2(x(ncoef1+ncoef2+ncoef3+1:ncoef1+ncoef2+ncoef3+ncoef4),S4,'db4') + waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5),S5,'db5') + waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6),S6,'db6') + waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7),S7,'db7'))/sqrt(wave_num);

otherwise
    Psi = @(x) [wavedec2(x,nlevel,'db1')'; wavedec2(x,nlevel,'db2')';wavedec2(x,nlevel,'db3')'; wavedec2(x,nlevel,'db4')'; wavedec2(x,nlevel,'db5')'; wavedec2(x,nlevel,'db6')'; wavedec2(x,nlevel,'db7')';wavedec2(x,nlevel,'db8')']/sqrt(wave_num); 
    Psit = @(x) (waverec2(x(1:ncoef1),S1,'db1') + waverec2(x(ncoef1+1:ncoef1+ncoef2),S2,'db2') + waverec2(x(ncoef1+ncoef2+1:ncoef1+ncoef2+ncoef3),S3,'db3') + waverec2(x(ncoef1+ncoef2+ncoef3+1:ncoef1+ncoef2+ncoef3+ncoef4),S4,'db4') + waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5),S5,'db5') + waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6),S6,'db6') + waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7),S7,'db7') + waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ ncoef7+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7+ncoef),S,'db8'))/sqrt(wave_num);

end


% create structual random measurements matrix A and At
% T. T. Do, et al. “Fast and efficient compressive sensing using structurally random matrices,” IEEE Transactions on Signal Processing, vol. 60, no. 1, pp. 139–154, Jan 2012.
imSize = size(img);
N=imSize(1)*imSize(2);
M=round(rate*N);

p=randperm(N);
mask=zeros(N,1);
mask(p(1:M))=1;
mask=reshape(mask,imSize);

%% Spread spectrum operator
% Mask
ind = find(mask==1);
% Masking matrix (sparse matrix in matlab)
Ma = sparse(1:numel(ind), ind, ones(numel(ind), 1), numel(ind), numel(img));
    
%Spread spectrum sequence
ss=rand(imSize);
CC=(2*(ss<0.5)-1);

A = @(x) Ma*reshape(dct2(CC.*x)/sqrt(numel(ind)), numel(x), 1);
At = @(x) CC.*(idct2(reshape(Ma'*x(:), imSize)*sqrt(numel(ind))));


% noisy measurements y
% note that for different noise levels, the regularization parameters need to be tuned accordingly for best performance
y=A(img);
noise = randn(size(y));
y=y+0.02*noise;

psnr_rec = zeros(1,7);
% L1 minimization
par.reg_fun='l1';
par.X0 = zeros(imSize);     % initialize the estimated image with all zeros
par.maxiter = 1000;         % maximum number of iterations
par.pval = 1;               % dummy parameter, does not play any role for l1 recovery here 
par.epsilon = eps;          % avoid division by 0
par.tol = 1e-6;             % convergence tolerance, adjust this accordingly
par.kappa = 2;              % should be twice the largest eigenvalue of (Psi*At*A*Psit), i.e. 2 in this case.
par.cri_type=1;             % choose convergence criterion

lambda = 0.1;               % the optimal lambda value needs to be tuned accordingly
xr_l1 = recovery_sara_l1_fista(y, A, At, lambda, Psi, Psit, par);
psnr_rec(1) = psnr(img, xr_l1);


% Lp minimization
par.reg_fun='lp';
% for best performance, use the solution from L1 minimization to initialize
par.X0=xr_l1;               % initialize with l1 recovery
par.maxiter = 1000;         % maximum number of iterations
par.pval = 0.5;             % the optimal p value needs to be tuned accordingly
par.epsilon = eps;          % avoid division by 0
par.tol = 1e-6;             % convergence tolerance, adjust this accordingly
par.kappa = 2;              % should be twice the largest eigenvalue of (Psi*At*A*Psit), i.e. 2 in this case.
par.cri_type=1;             % choose convergence criterion

lambda = 10;              % the optimal lambda value needs to be tuned accordingly
xr_lp = recovery_sara_fista(y, A, At, lambda, Psi, Psit, par);
psnr_rec(2) = psnr(img, xr_lp);


% Shannon entropy function minimization
par.reg_fun='shannon_ef';
% for best performance, use the solution from L1 minimization to initialize
par.X0=xr_l1;               % initialize with l1 recovery
par.maxiter = 1000;         % maximum number of iterations
par.pval = 0.8;               % the optimal p value needs to be tuned accordingly
par.epsilon = eps;          % avoid division by 0
par.tol = 1e-6;             % convergence tolerance, adjust this accordingly
par.kappa = 2;              % should be twice the largest eigenvalue of (Psi*At*A*Psit), i.e. 2 in this case.
par.cri_type=1;             % choose convergence criterion

lambda = 1000000;              % the optimal lambda value needs to be tuned accordingly
xr_shannon_ef = recovery_sara_fista(y, A, At, lambda, Psi, Psit, par);
psnr_rec(3) = psnr(img, xr_shannon_ef);


% Renyi entropy function minimization
par.reg_fun='renyi_ef';
% for best performance, use the solution from L1 minimization to initialize
par.X0=xr_l1;               % initialize with l1 recovery
par.maxiter = 1000;         % maximum number of iterations
par.pval = 0.9;             % the optimal p value needs to be tuned accordingly
par.alpha = 0.8;            % the optimal alpha value needs to be tuned accordingly
par.epsilon = eps;          % avoid division by 0
par.tol = 1e-6;             % convergence tolerance, adjust this accordingly
par.kappa = 2;              % should be twice the largest eigenvalue of (Psi*At*A*Psit), i.e. 2 in this case.
par.cri_type=1;             % choose convergence criterion

lambda=1000000;               % the optimal lambda value needs to be tuned accordingly
xr_renyi_ef = recovery_sara_fista(y, A, At, lambda, Psi, Psit, par);
psnr_rec(4) = psnr(img, xr_renyi_ef);


% L_1-L_infinity function minimization
par.reg_fun='l1_linfinity';
% for best performance, use the solution from L1 minimization to initialize
par.X0=xr_l1;               % initialize with l1 recovery
par.maxiter = 1000;         % maximum number of iterations
par.pval = 1;               % dummy variable to be removed in a later version
par.epsilon = eps;          % avoid division by 0
par.tol = 1e-6;             % convergence tolerance, adjust this accordingly
par.kappa = 2;              % should be twice the largest eigenvalue of (Psi*At*A*Psit), i.e. 2 in this case.
par.cri_type=1;             % choose convergence criterion

lambda=1;       % the optimal lambda value needs to be tuned accordingly
xr_l1_linfinity = recovery_sara_l1_linfinity_fista(y, A, At, lambda, Psi, Psit, par);
psnr_rec(5) = psnr(img, xr_l1_linfinity);



% logarithm of energy minimization via regularized FOCUSS algorithm
par.reg_fun='log_nrg';
% for best performance, use the solution from L1 minimization to initialize
par.X0=xr_l1;               % initialize with l1 recovery
par.maxiter = 1000;         % maximum number of iterations
par.pval = 1;               % dummy variable to be removed in a later version
par.epsilon = eps;          % avoid division by 0
par.tol=1e-6;               % convergence tolerance, adjust this accordingly
par.kappa = 2;              % should be twice the largest eigenvalue of (Psi*At*A*Psit), i.e. 2 in this case.
par.cri_type=1;             % choose convergence criterion

lambda=50;                 % the optimal lambda value needs to be tuned accordingly
xr_log_nrg = recovery_sara_log_nrg_fista(y, A, At, lambda, Psi, Psit, par);
psnr_rec(6) = psnr(img, xr_log_nrg);

% iterative hard-thresholding
% The de facto operator norm of A is bounded above by 1, convergence of IHT is thus guaranteed
par.reg_fun='iht';
% for best performance, use the solution from L1 minimization to initialize
par.X0=xr_l1;               % initialize with l1 recovery
par.maxiter = 1000;         % maximum number of iterations
par.pval = 1;               % dummy variable to be removed in a later version
par.epsilon = eps;          % avoid division by 0
par.tol=1e-6;               % convergence tolerance, adjust this accordingly
par.kappa = 2;              % should be twice the largest eigenvalue of (Psi*At*A*Psit), i.e. 2 in this case.
par.cri_type=1;             % choose convergence criterion

lambda=1e-3;                   % the optimal lambda value needs to be tuned accordingly
xr_iht = recovery_sara_iht_fista(y, A, At, lambda, Psi, Psit, par);
psnr_rec(7) = psnr(img, xr_iht);



fprintf('PSNR(dB) of recovered images with a sampling rate of %.2f:\n', rate)
fprintf('L1  : %.2f dB\n', psnr_rec(1))
fprintf('Lp  : %.2f dB\n', psnr_rec(2))
fprintf('SEF : %.2f dB\n', psnr_rec(3))
fprintf('REF : %.2f dB\n', psnr_rec(4))
fprintf('L1-Linfinity : %.2f dB\n', psnr_rec(5))
fprintf('Log-NRG : %.2f dB\n', psnr_rec(6))
fprintf('IHT : %.2f dB\n', psnr_rec(7))


