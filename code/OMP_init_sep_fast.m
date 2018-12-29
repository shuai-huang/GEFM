function [x,r,normR] = OMP_init_sep_fast( A_one, A_two, b, k_one, k_two, opts)
% modified to handle initialization and SRC by S. H.

% x = OMP( A, b, k )
%   uses the Orthogonal Matching Pursuit algorithm (OMP)
%   to estimate the solution to the equation
%       b = A*x     (or b = A*x + noise )
%   where there is prior information that x is sparse.
%
%   "A" may be a matrix, or it may be a cell array {Af,At}
%   where Af and At are function handles that compute the forward and transpose
%   multiplies, respectively.
%
% [x,r,normR,residHist,errHist] = OMP( A, b, k, errFcn, opts )
%   is the full version.
% Outputs:
%   'x' is the k-sparse estimate of the unknown signal
%   'r' is the residual b - A*x
%   'normR' = norm(r)
%   'residHist'     is a vector with normR from every iteration
%   'errHist'       is a vector with the outout of errFcn from every iteration
%
% Inputs:
%   'A'     is the measurement matrix
%   'b'     is the vector of observations
%   'k'     is the estimate of the sparsity (you may wish to purposefully
%              over- or under-estimate the sparsity, depending on noise)
%              N.B. k < size(A,1) is necessary, otherwise we cannot
%                   solve the internal least-squares problem uniquely.
%
%   'k' (alternative usage):
%           instead of specifying the expected sparsity, you can specify
%           the expected residual. Set 'k' to the residual. The code
%           will automatically detect this if 'k' is not an integer;
%           if the residual happens to be an integer, so that confusion could
%           arise, then specify it within a cell, like {k}.
%
%   'errFcn'    (optional; set to [] to ignore) is a function handle
%              which will be used to calculate the error; the output
%              should be a scalar
%
%   'opts'  is a structure with more options, including:
%       .printEvery = is an integer which controls how often output is printed
%       .maxiter    = maximum number of iterations
%       .slowMode   = whether to compute an estimate at every iteration
%                       This computation is slower, but it allows you to
%                       display the error at every iteration (via 'errFcn')
%
%       Note that these field names are case sensitive!
%
% If you need a faster implementation, try the very good C++ implementation
% (with mex interface to Matlab) in the "SPAMS" toolbox, available at:
%   http://www.di.ens.fr/willow/SPAMS/
% The code in SPAMS is precompiled for most platforms, so it is easy to install.
% SPAMS uses Cholesky decompositions and uses a slightly different
%   updating rule to select the next atom.
%
% Stephen Becker, Aug 1 2011.  srbecker@alumni.caltech.edu
% Updated Dec 12 2012, fixing bug for complex data, thanks to Noam Wagner.
%   See also CoSaMP, test_OMP_and_CoSaMP


A= [A_one A_two]; 

% What stopping criteria to use? either a fixed # of iterations,
%   or a desired size of residual:
target_resid    = 1e-6;
k=k_one + k_two;	% the combined sparsity upperbound

if iscell(A)
    LARGESCALE  = true;
    Af  = A{1};
    At  = A{2};     % we don't really need this...
else
    LARGESCALE  = false;
    Af  = @(x) A*x;
    At  = @(x) A'*x;
end

% -- Intitialize --
% start at x = 0, so r = b - A*x = b
r           = b-A*opts.X0;
normR       = norm(r);
Ar          = At(r);
N           = size(Ar,1);       % number of atoms
M           = size(r,1);        % size of atoms
if k > M
    error('K cannot be larger than the dimension of the atoms');
end
unitVector  = zeros(N,1);
x           = opts.X0;

idx_tmp = 1:length(x);
indx_set    = idx_tmp(x~=0);

idx_tmp_one = idx_tmp(1:size(A_one,2));
x_one = x(1:size(A_one,2));
indx_set_one = idx_tmp_one(x_one~=0);

idx_tmp_two = idx_tmp((size(A_one,2)+1):end);
x_two = x((size(A_one,2)+1):end);
indx_set_two = idx_tmp_two(x_two~=0);

A_T_nonorth         = zeros(M,k);
A_T_nonorth(:,1:length(indx_set)) = A(:,indx_set);

A_T = zeros(M,k);
A_T(:,1:length(indx_set(indx_set>=(size(A_one,2)+1)))) = A(:,indx_set(indx_set>=(size(A_one,2)+1)));
noise_idx_init = indx_set(indx_set>=(size(A_one,2)+1))-size(A_one,2);
num_dict_noise_init = length(indx_set(indx_set>=(size(A_one,2)+1)));
num_dict = length(indx_set(indx_set>=(size(A_one,2)+1)));
% orthonormalize the first size(A_one,2) columns
for (col_idx=indx_set)
    if (col_idx<=size(A_one,2))
        A_T_atom = A(:, col_idx);
        A_T_atom(noise_idx_init)=0;
        if (num_dict>num_dict_noise_init)
        for (j=(num_dict_noise_init+1):num_dict)
            A_T_atom = A_T_atom - (A_T(:,j)'*A_T_atom)*A_T(:,j);
        end
        end
        A_T_atom = A_T_atom/norm(A_T_atom);
        A_T(:,num_dict+1)=A_T_atom;
        num_dict=num_dict+1;
    end
end
%A_T(:,1:length(indx_set)) = orth(A(:,indx_set));

kk_init = length(indx_set);

if (k>length(indx_set))
    indx_set = [indx_set repmat(0, 1, k-length(indx_set))];
end
%if (k_one>length(idx_set_one))
%    indx_set_one = [indx_set_one repmat(0, 1, k_one-length(indx_set_one))];
%end
%if (k_two>length(idx_set_two))
%    indx_set_two = [indx_set_two repmat(0, 1, k_two-length(indx_set_two))];
%end

fprintf('Idx_one init: %d\n', length(indx_set_one));
fprintf('Idx_two init: %d\n', length(indx_set_two));

x_T=[];

for kk = kk_init+1:k
    
    % -- Step 1: find new index and atom to add
    [dummy,ind_new]     = max(abs(Ar));
    if (ind_new<=size(A_one,2))
        if (length(indx_set_one(indx_set_one~=0))<k_one)
            indx_set_one = [indx_set_one ind_new];
        else
            [dummy, ind_new] = max(abs(Ar((size(A_one,2)+1):end)));
            ind_new = ind_new + size(A_one,2);
            indx_set_two = [indx_set_two ind_new];
        end
    else
        if (length(indx_set_two(indx_set_two~=0))<k_two)
            indx_set_two = [indx_set_two ind_new];
        else
            [dummy, ind_new] = max(abs(Ar(1:size(A_one,2))));
            indx_set_one = [indx_set_one ind_new];
        end
    end
    
    indx_set(kk)    = ind_new;
    
    if LARGESCALE
        unitVector(ind_new)     = 1;
        atom_new                = Af( unitVector );
        unitVector(ind_new)     = 0;
    else
        atom_new    = A(:,ind_new);
    end
    
    A_T_nonorth(:,kk)   = atom_new;     % before orthogonalizing and such

    % -- Step 2: update residual

    %x_T = A_T_nonorth(:,1:kk)\b;
    %x( indx_set(1:kk) )   = x_T;
    %r = b - A_T_nonorth(:,1:kk)*x_T;

    % First, orthogonalize 'atom_new' against all previous atoms
    % We use MGS
    %atom_new = atom_new - sum(atom_new(noise_idx_init));
    atom_new(noise_idx_init)=0;
    for j = (num_dict_noise_init+1):(kk-1)
         %             atom_new    = atom_new - (atom_new'*A_T(:,j))*A_T(:,j);
        % Thanks to Noam Wagner for spotting this bug. The above line
        % is wrong when the data is complex. Use this:
        atom_new    = atom_new - (A_T(:,j)'*atom_new)*A_T(:,j);
    end
    % Second, normalize:
    atom_new        = atom_new/norm(atom_new);
    A_T(:,kk)       = atom_new;
    % Third, solve least-squares problem (which is now very easy
    %   since A_T(:,1:kk) is orthogonal )
    x_T     = A_T(:,1:kk)'*b;
    x( indx_set(1:kk) )   = x_T;      % note: indx_set is guaranteed to never shrink
    % Fourth, update residual:
    %     r       = b - Af(x); % wrong!
    r       = b - A_T(:,1:kk)*x_T;

    
    normR = norm(r);
    %fprintf('Iteration %d: %f\t%d\t%d\n', kk, normR, length(indx_set_one), length(indx_set_two))

    if normR < target_resid
        fprintf('Residual reached desired size (%.2e < %.2e)\n', normR, target_resid );
        break;
    end
    
    if kk < k
        Ar  = At(r); % prepare for next round
    end
    
end

if (target_resid) && ( normR >= target_resid )
    fprintf('Warning: did not reach target size of residual\n');
end

r       = b - A_T_nonorth(1:kk)*x_T;
normR   = norm(r);

end % end of main function
