% add the path, change this if needed
addpath(genpath('./code'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Processing data %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%


% read training and test data
load('CroppedYale_96_84_2414_subset.mat')
class_train = facecls((facesubset==1)|(facesubset==2)); % the training labels should be strictly arranged from 1 to C
class_test = facecls(facesubset==3);
faces = double(faces);
num_class=length(unique(class_train));

face_mat=zeros(96*84,2414);
for (i=1:2414)
    face_tmp = squeeze(faces(i,:,:));
    face_mat(:,i) = face_tmp(:);
end

face_train = face_mat(:,(facesubset==1)|(facesubset==2));
face_test = face_mat(:,facesubset==3);


% normalize the training faces
for (i=1:size(face_train, 2)) 
    face_seq_tmp = face_train(:,i);
    face_train(:,i) = face_seq_tmp/norm(face_seq_tmp, 'fro');
end


% corrupt the testing faces, replace random pixels with values from 0 ~ 255
% 90% of the pixels will be corrupted
corr_rate = 0.9;

rng(1); % set the random number generator seed
for (i = 1:size(face_test,2))
    face_seq_tmp = face_test(:,i);

    ind = randperm(length(face_seq_tmp));
    indice = ind(1 : ceil(length(face_seq_tmp)*corr_rate)); 
    face_seq_tmp(indice) = randi(256, size(indice)) - 1;

    face_test(:,i)=face_seq_tmp/norm(face_seq_tmp, 'fro');
end


% add the dictionary for the random noise, i.e. a identity matrix
noise_dict = diag(ones(size(face_train,1),1));

%%%%%%%%%%%%%%%%%%
%%%%% l1 SRC %%%%%
%%%%%%%%%%%%%%%%%%


% function parameters
Par.tol = 1e-6;         % stopping criterion of the FISTA algorithm, make this smaller to improve accuracy
Par.maxiter = 1000;     % the maximum number of iterations of the FISTA algorithm in the main loop
Par.innermaxiter = 1;   % the maximum number of iterations in the inner loop
Par.epsilon = 1e-12;    % add a smalle number to avoid 1/0

% start sparse signal recovery
% L1 minimization
Par.X0 = zeros(size(face_train,2)+size(noise_dict,2), 1);           % initialization
Par.p = 1;	                    % choose a value around 1, needs to be tuned
lambda = 1e-3;          % regularization parameter for the training dictionary, needs to be *tuned* for different applications
mu = 2.5;                % lambda*mu is the regularization parameter for the noise dictionary, needs to be *tuned* for different applications

% compute the Lipschitz constant
[U, S_diag, V] = eigs([face_train noise_dict]'*[face_train noise_dict]);
Par.kappa = 2*max(diag(S_diag));    % the Lipschitz constant


xr_test_l1=[];
val_test=[];
idx=1:size(face_train,2);

for (i=1:size(face_test,2))
    fprintf('Computing %d -th sample\n', i)

    xr = ssr_l1_sep(face_test(:,i), face_train, noise_dict, Par, lambda, mu);
    xr_test_l1=[xr_test_l1 xr];

    xr_face_train = xr(1:size(face_train,2));
    xr_noise_dict = xr(size(face_train,2)+1:end);

    res=face_test(:,i) - noise_dict * xr_noise_dict;
    res_val=sum(res.^2);
    val_test_tmp=[];
    for (j=1:num_class)
        idx_tmp=idx(class_train==j);
        if (sum(abs(xr_face_train(idx_tmp,1)))~=0)
            res_tmp=res-face_train(:,idx_tmp)*xr_face_train(idx_tmp,1);
            val_test_tmp=[val_test_tmp sum(res_tmp.^2)];
        else
            val_test_tmp=[val_test_tmp res_val];
        end
    end 
    val_test=[val_test; val_test_tmp];
end

% use SRC to find the test label
lab_test_l1=[];
for (i=1:size(face_test,2))
    val_test_tmp=val_test(i,:);
    [val_test_tmp_min, val_test_tmp_idx]=min(val_test_tmp);
    lab_test_l1=[lab_test_l1 val_test_tmp_idx];
end



%%%%%%%%%%%%%%%%%%
%%%%% lp SRC %%%%%
%%%%%%%%%%%%%%%%%%

clear Par;
% function parameters
Par.tol = 1e-6;         % stopping criterion of the FISTA algorithm, make this smaller to improve accuracy
Par.maxiter = 1000;     % the maximum number of iterations of the FISTA algorithm in the main loop
Par.innermaxiter = 1;   % the maximum number of iterations in the inner loop
Par.epsilon = 1e-12;    % add a smalle number to avoid 1/0

% start sparse signal recovery
% Lp minimization
Par.p = 0.5;	                    % choose a value around 1, needs to be tuned
lambda = 1e-4;          % regularization parameter for the training dictionary, needs to be *tuned* for different applications
mu = 5;                % lambda*mu is the regularization parameter for the noise dictionary, needs to be *tuned* for different applications

% compute the Lipschitz constant
[U, S_diag, V] = eigs([face_train noise_dict]'*[face_train noise_dict]);
Par.kappa = 2*max(diag(S_diag));    % the Lipschitz constant

% for best performance, use the solution from L1 minimization to initialize
% Attention: for higher efficientcy and speed, it is recommended using a computing cluster to recognize test images in parallel
xr_test_lp=[];
val_test=[];
idx=1:size(face_train,2);

for (i=1:size(face_test,2))
    fprintf('Computing %d -th sample\n', i)
    Par.X0 = xr_test_l1(:,i);

    xr = ssr_lp_sep(face_test(:,i), face_train, noise_dict, Par, lambda, mu);
    xr_test_lp=[xr_test_lp xr];

    xr_face_train = xr(1:size(face_train,2));
    xr_noise_dict = xr(size(face_train,2)+1:end);

    res=face_test(:,i) - noise_dict * xr_noise_dict;
    res_val=sum(res.^2);
    val_test_tmp=[];
    for (j=1:num_class)
        idx_tmp=idx(class_train==j);
        if (sum(abs(xr_face_train(idx_tmp,1)))~=0)
            res_tmp=res-face_train(:,idx_tmp)*xr_face_train(idx_tmp,1);
            val_test_tmp=[val_test_tmp sum(res_tmp.^2)];
        else
            val_test_tmp=[val_test_tmp res_val];
        end
    end 
    val_test=[val_test; val_test_tmp];
end


% use SRC to find the test label
lab_test_lp=[];
for (i=1:size(face_test,2))
    val_test_tmp=val_test(i,:);
    [val_test_tmp_min, val_test_tmp_idx]=min(val_test_tmp);
    lab_test_lp=[lab_test_lp val_test_tmp_idx];
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Shannon entropy function SRC %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear Par;
% function parameters
Par.tol = 1e-6;         % stopping criterion of the FISTA algorithm, make this smaller to improve accuracy
Par.maxiter = 1000;     % the maximum number of iterations of the FISTA algorithm in the main loop
Par.innermaxiter = 1;   % the maximum number of iterations in the inner loop
Par.epsilon = 1e-12;    % add a smalle number to avoid 1/0

% The following three parameters should be tuned for best performance
Par.p = 1.2;	        % choose a value around 1, needs to be *tuned* for different applications
lambda = 1e-3;          % regularization parameter for the training dictionary, needs to be *tuned* for different applications
mu = 75;                % lambda*mu is the regularization parameter for the noise dictionary, needs to be *tuned* for different applications

% compute the Lipschitz constant
[U, S_diag, V] = eigs([face_train noise_dict]'*[face_train noise_dict]);
Par.kappa = 2*max(diag(S_diag));    % the Lipschitz constant


% start sparse signal recovery
% for best performance, use the solution from L1 minimization to initialize
xr_test_shannon=[];
val_test=[];
idx=1:size(face_train,2);

% for best performance, use the solution from L1 minimization to initialize
% Attention: for higher efficientcy and speed, it is recommended using a computing cluster to recognize test images in parallel
for (i=1:size(face_test,2))
    fprintf('Computing %d -th sample\n', i)
    Par.X0 = xr_test_l1(:,i);

    xr = ssr_shannon_ef_sep(face_test(:,i), face_train, noise_dict, Par, lambda, mu);
    xr_test_shannon=[xr_test_shannon xr];
    
    xr_face_train = xr(1:size(face_train,2));
    xr_noise_dict = xr(size(face_train,2)+1:end);

    res=face_test(:,i) - noise_dict * xr_noise_dict;
    res_val=sum(res.^2);
    val_test_tmp=[];
    for (j=1:num_class)
        idx_tmp=idx(class_train==j);
        if (sum(abs(xr_face_train(idx_tmp,1)))~=0)
            res_tmp=res-face_train(:,idx_tmp)*xr_face_train(idx_tmp,1);
            val_test_tmp=[val_test_tmp sum(res_tmp.^2)];
        else
            val_test_tmp=[val_test_tmp res_val];
        end
    end 
    val_test=[val_test; val_test_tmp];
end

% use SRC to find the test label
lab_test_shannon=[];
for (i=1:size(face_test,2))
    val_test_tmp=val_test(i,:);
    [val_test_tmp_min, val_test_tmp_idx]=min(val_test_tmp);
    lab_test_shannon=[lab_test_shannon val_test_tmp_idx];
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Renyi entropy function SRC %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear Par;
% function parameters
Par.tol = 1e-6;         % stopping criterion of the FISTA algorithm, make this smaller to improve accuracy
Par.maxiter = 1000;     % the maximum number of iterations of the FISTA algorithm in the main loop
Par.epsilon = 1e-12;    % add a smalle number to avoid 1/0

% The following three parameters should be tuned for best performance
Par.p = 1.15;	        % choose a value around 1, needs to be *tuned* for different applications
Par.alpha = 1.1;
lambda = 1e-3;          % regularization parameter for the training dictionary, needs to be *tuned* for different applications
mu = 75;                % lambda*mu is the regularization parameter for the noise dictionary, needs to be *tuned* for different applications

% compute the Lipschitz constant
[U, S_diag, V] = eigs([face_train noise_dict]'*[face_train noise_dict]);
Par.kappa = 2*max(diag(S_diag));    % the Lipschitz constant

% for best performance, use the solution from L1 minimization to initialize
xr_test_renyi=[];
val_test=[];
idx=1:size(face_train,2);

% for best performance, use the solution from L1 minimization to initialize
% Attention: for higher efficientcy and speed, it is recommended using a computing cluster to recognize test images in parallel
for (i=1:size(face_test,2))
    fprintf('Computing %d -th sample\n', i)
    Par.X0 = xr_test_l1(:,i);

    xr = ssr_renyi_ef_sep(face_test(:,i), face_train, noise_dict, Par, lambda, mu);
    xr_test_renyi=[xr_test_renyi xr];

    xr_face_train = xr(1:size(face_train,2));
    xr_noise_dict = xr(size(face_train,2)+1:end);

    res=face_test(:,i) - noise_dict * xr_noise_dict;
    res_val=sum(res.^2);
    val_test_tmp=[];
    for (j=1:num_class)
        idx_tmp=idx(class_train==j);
        if (sum(abs(xr_face_train(idx_tmp,1)))~=0)
            res_tmp=res-face_train(:,idx_tmp)*xr_face_train(idx_tmp,1);
            val_test_tmp=[val_test_tmp sum(res_tmp.^2)];
        else
            val_test_tmp=[val_test_tmp res_val];
        end
    end 
    val_test=[val_test; val_test_tmp];
end

% use SRC to find the test label
lab_test_renyi=[];
for (i=1:size(face_test,2))
    val_test_tmp=val_test(i,:);
    [val_test_tmp_min, val_test_tmp_idx]=min(val_test_tmp);
    lab_test_renyi=[lab_test_renyi val_test_tmp_idx];
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% L_1-L_infinity SRC %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear Par;
% function parameters
Par.tol = 1e-6;         % stopping criterion of the FISTA algorithm, make this smaller to improve accuracy
Par.maxiter = 1000;     % the maximum number of iterations of the FISTA algorithm in the main loop
Par.innermaxiter = 1;   % the maximum number of iterations in the inner loop
Par.epsilon = 1e-12;    % add a smalle number to avoid 1/0

% start sparse signal recovery
% Lp minimization
Par.p = 0.5;	                    % choose a value around 1, needs to be tuned
lambda = 1e-6;          % regularization parameter for the training dictionary, needs to be *tuned* for different applications
mu = 2.5;                % lambda*mu is the regularization parameter for the noise dictionary, needs to be *tuned* for different applications

% compute the Lipschitz constant
[U, S_diag, V] = eigs([face_train noise_dict]'*[face_train noise_dict]);
Par.kappa = 2*max(diag(S_diag));    % the Lipschitz constant

% for best performance, use the solution from L1 minimization to initialize
% Attention: for higher efficientcy and speed, it is recommended using a computing cluster to recognize test images in parallel
xr_test_l1_linfinity=[];
val_test=[];
idx=1:size(face_train,2);

for (i=1:size(face_test,2))
    fprintf('Computing %d -th sample\n', i)
    Par.X0 = xr_test_l1(:,i);

    xr = ssr_l1_linfinity_sep(face_test(:,i), face_train, noise_dict, Par, lambda, mu);
    xr_test_l1_linfinity=[xr_test_l1_linfinity xr];

    xr_face_train = xr(1:size(face_train,2));
    xr_noise_dict = xr(size(face_train,2)+1:end);

    res=face_test(:,i) - noise_dict * xr_noise_dict;
    res_val=sum(res.^2);
    val_test_tmp=[];
    for (j=1:num_class)
        idx_tmp=idx(class_train==j);
        if (sum(abs(xr_face_train(idx_tmp,1)))~=0)
            res_tmp=res-face_train(:,idx_tmp)*xr_face_train(idx_tmp,1);
            val_test_tmp=[val_test_tmp sum(res_tmp.^2)];
        else
            val_test_tmp=[val_test_tmp res_val];
        end
    end 
    val_test=[val_test; val_test_tmp];
end


% use SRC to find the test label
lab_test_l1_linfinity=[];
for (i=1:size(face_test,2))
    val_test_tmp=val_test(i,:);
    [val_test_tmp_min, val_test_tmp_idx]=min(val_test_tmp);
    lab_test_l1_linfinity=[lab_test_l1_linfinity val_test_tmp_idx];
end




%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Log-NRG   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

clear Par;
% function parameters
Par.tol = 1e-6;         % stopping criterion of the FISTA algorithm, make this smaller to improve accuracy
Par.maxiter = 100;     % the maximum number of iterations of the FISTA algorithm in the main loop
lambda = 1e-6;          % regularization parameter for the training dictionary, needs to be *tuned* for different applications
mu = 5;                % lambda*mu is the regularization parameter for the noise dictionary, needs to be *tuned* for different applications

% compute the Lipschitz constant
[U, S_diag, V] = eigs([face_train noise_dict]'*[face_train noise_dict]);
Par.kappa = 2*max(diag(S_diag));    % the Lipschitz constant


% for best performance, use the solution from L1 minimization to initialize
xr_test_log_nrg=[];
val_test=[];
idx=1:size(face_train,2);

% for best performance, use the solution from L1 minimization to initialize
% Attention: for higher efficientcy and speed, it is recommended using a computing cluster to recognize test images in parallel
for (i=1:size(face_test,2))
    fprintf('Computing %d -th sample\n', i)
    Par.X0 = xr_test_l1(:,i);

    xr = focuss_sep(face_test(:,i), face_train, noise_dict, -1, false, Par.X0, Par.maxiter, 0, lambda, mu);
    xr_test_log_nrg=[xr_test_log_nrg xr];

    xr_face_train = xr(1:size(face_train,2));
    xr_noise_dict = xr(size(face_train,2)+1:end);

    res=face_test(:,i) - noise_dict * xr_noise_dict;
    res_val=sum(res.^2);
    val_test_tmp=[];
    for (j=1:num_class)
        idx_tmp=idx(class_train==j);
        if (sum(abs(xr_face_train(idx_tmp,1)))~=0)
            res_tmp=res-face_train(:,idx_tmp)*xr_face_train(idx_tmp,1);
            val_test_tmp=[val_test_tmp sum(res_tmp.^2)];
        else
            val_test_tmp=[val_test_tmp res_val];
        end
    end 
    val_test=[val_test; val_test_tmp];
end

% use SRC to find the test label
lab_test_log_nrg=[];
for (i=1:size(face_test,2))
    val_test_tmp=val_test(i,:);
    [val_test_tmp_min, val_test_tmp_idx]=min(val_test_tmp);
    lab_test_log_nrg=[lab_test_log_nrg val_test_tmp_idx];
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Iterative Hard-thresholding   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear Par;

% function parameters
Par.tol = 1e-6;         % stopping criterion of the FISTA algorithm, make this smaller to improve accuracy
Par.maxiter = 1000;     % the maximum number of iterations of the FISTA algorithm in the main loop
lambda = 1e-3;          % regularization parameter for the training dictionary, needs to be *tuned* for different applications
mu = 6;                % lambda*mu is the regularization parameter for the noise dictionary, needs to be *tuned* for different applications

% compute the Lipschitz constant
[U, S_diag, V] = eigs([face_train noise_dict]'*[face_train noise_dict]);
Par.kappa = 2*max(diag(S_diag));    % the Lipschitz constant

face_train_normalized = face_train/sqrt(max(S_diag));
noise_dict_normalized = noise_dict/sqrt(max(S_diag));
face_test_normalized = face_test/sqrt(max(S_diag));

% for best performance, use the solution from L1 minimization to initialize
xr_test_iht=[];
val_test=[];
idx=1:size(face_train_normalized,2);

% for best performance, use the solution from L1 minimization to initialize
% Attention: for higher efficientcy and speed, it is recommended using a computing cluster to recognize test images in parallel
for (i=1:size(face_test_normalized,2))
    fprintf('Computing %d -th sample\n', i)
    Par.X0 = xr_test_l1(:,i);

    xr = ssr_iht_sep(face_test_normalized(:,i), face_train_normalized, noise_dict_normalized, Par, lambda, mu);
    xr_test_iht=[xr_test_iht xr];

    xr_face_train = xr(1:size(face_train,2));
    xr_noise_dict = xr(size(face_train,2)+1:end);

    res=face_test(:,i) - noise_dict * xr_noise_dict;
    res_val=sum(res.^2);
    val_test_tmp=[];
    for (j=1:num_class)
        idx_tmp=idx(class_train==j);
        if (sum(abs(xr_face_train(idx_tmp,1)))~=0)
            res_tmp=res-face_train(:,idx_tmp)*xr_face_train(idx_tmp,1);
            val_test_tmp=[val_test_tmp sum(res_tmp.^2)];
        else
            val_test_tmp=[val_test_tmp res_val];
        end
    end 
    val_test=[val_test; val_test_tmp];
end

% use SRC to find the test label
lab_test_iht=[];
for (i=1:size(face_test,2))
    val_test_tmp=val_test(i,:);
    [val_test_tmp_min, val_test_tmp_idx]=min(val_test_tmp);
    lab_test_iht=[lab_test_iht val_test_tmp_idx];
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Orthogonal matching pursuit   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear Par;

% function parameters
sparsity_one = 20;
sparsity_two = 7000;

% for best performance, use the solution from L1 minimization to initialize
xr_test_omp=[];
val_test=[];
idx=1:size(face_train,2);

% for best performance, use the solution from L1 minimization to initialize
% Attention: for higher efficientcy and speed, it is recommended using a computing cluster to recognize test images in parallel
for (i=1:size(face_test,2))
    fprintf('Computing %d -th sample\n', i)
    omp_opts.X0 = xr_test_l1(:,i);
    omp_opts.maxiter = 1000;

    xr = OMP_init_sep_fast(face_train, noise_dict, face_test(:,i), sparsity_one, sparsity_two, omp_opts);
    xr_test_omp=[xr_test_omp xr];

    xr_face_train = xr(1:size(face_train,2));
    xr_noise_dict = xr(size(face_train,2)+1:end);

    res=face_test(:,i) - noise_dict * xr_noise_dict;
    res_val=sum(res.^2);
    val_test_tmp=[];
    for (j=1:num_class)
        idx_tmp=idx(class_train==j);
        if (sum(abs(xr_face_train(idx_tmp,1)))~=0)
            res_tmp=res-face_train(:,idx_tmp)*xr_face_train(idx_tmp,1);
            val_test_tmp=[val_test_tmp sum(res_tmp.^2)];
        else
            val_test_tmp=[val_test_tmp res_val];
        end
    end 
    val_test=[val_test; val_test_tmp];
end

% use SRC to find the test label
lab_test_omp=[];
for (i=1:size(face_test,2))
    val_test_tmp=val_test(i,:);
    [val_test_tmp_min, val_test_tmp_idx]=min(val_test_tmp);
    lab_test_omp=[lab_test_omp val_test_tmp_idx];
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   CoSaMP   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear Par;

% function parameters
sparsity_one = 20;
sparsity_two = 7000;

% for best performance, use the solution from L1 minimization to initialize
xr_test_cosamp=[];
val_test=[];
idx=1:size(face_train,2);

% for best performance, use the solution from L1 minimization to initialize
% Attention: for higher efficientcy and speed, it is recommended using a computing cluster to recognize test images in parallel
for (i=1:size(face_test,2))
    fprintf('Computing %d -th sample\n', i)
    cosamp_opts.X0 = xr_test_l1(:,i);
    cosamp_opts.maxiter = 100;
    cosamp_opts.normTol = 1e-6;
    cosamp_opts.support_tol = 1e-6;

    xr = CoSaMP_init_sep_fast(face_train, noise_dict, face_test(:,i), sparsity_one, sparsity_two, cosamp_opts);
    xr_test_cosamp=[xr_test_cosamp xr];

    xr_face_train = xr(1:size(face_train,2));
    xr_noise_dict = xr(size(face_train,2)+1:end);

    res=face_test(:,i) - noise_dict * xr_noise_dict;
    res_val=sum(res.^2);
    val_test_tmp=[];
    for (j=1:num_class)
        idx_tmp=idx(class_train==j);
        if (sum(abs(xr_face_train(idx_tmp,1)))~=0)
            res_tmp=res-face_train(:,idx_tmp)*xr_face_train(idx_tmp,1);
            val_test_tmp=[val_test_tmp sum(res_tmp.^2)];
        else
            val_test_tmp=[val_test_tmp res_val];
        end
    end 
    val_test=[val_test; val_test_tmp];
end

% use SRC to find the test label
lab_test_cosamp=[];
for (i=1:size(face_test,2))
    val_test_tmp=val_test(i,:);
    [val_test_tmp_min, val_test_tmp_idx]=min(val_test_tmp);
    lab_test_cosamp=[lab_test_cosamp val_test_tmp_idx];
end


fprintf('Face recognition via SRC :\n')
fprintf('Corruption rate : %d%%\n\n', corr_rate*100)
fprintf('L1  : %.2f%%\n', 100*length(lab_test_l1(lab_test_l1==class_test'))/length(class_test))
fprintf('Lp  : %.2f%%\n', 100*length(lab_test_lp(lab_test_lp==class_test'))/length(class_test))
fprintf('SEF : %.2f%%\n', 100*length(lab_test_shannon(lab_test_shannon==class_test'))/length(class_test))
fprintf('REF : %.2f%%\n', 100*length(lab_test_renyi(lab_test_renyi==class_test'))/length(class_test))
fprintf('L_1-L_infinity : %.2f%%\n', 100*length(lab_test_l1_linfinity(lab_test_l1_linfinity==class_test'))/length(class_test))
fprintf('Log-NRG : %.2f%%\n', 100*length(lab_test_log_nrg(lab_test_log_nrg==class_test'))/length(class_test))
fprintf('IHT : %.2f%%\n', 100*length(lab_test_iht(lab_test_iht==class_test'))/length(class_test))
fprintf('OMP : %.2f%%\n', 100*length(lab_test_omp(lab_test_omp==class_test'))/length(class_test))
fprintf('CoSaMP : %.2f%%\n', 100*length(lab_test_cosamp(lab_test_cosamp==class_test'))/length(class_test))

