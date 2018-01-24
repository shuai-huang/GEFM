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
Par.maxiter = 5000;     % the maximum number of iterations of the FISTA algorithm in the main loop
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


% for best performance, use the solution from L1 minimization to initialize
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
Par.maxiter = 5000;     % the maximum number of iterations of the FISTA algorithm in the main loop
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
Par.maxiter = 5000;     % the maximum number of iterations of the FISTA algorithm in the main loop
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


% for higher efficientcy and speed, it is recommended using a computing cluster to recognize test images in parallel
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
%%%%% Renyi entropy function SRC %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear Par;
% function parameters
Par.tol = 1e-6;         % stopping criterion of the FISTA algorithm, make this smaller to improve accuracy
Par.maxiter = 5000;     % the maximum number of iterations of the FISTA algorithm in the main loop
Par.innermaxiter = 1;   % the maximum number of iterations in the inner loop
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


% for higher efficientcy and speed, it is recommended using a computing cluster to recognize test images in parallel
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



