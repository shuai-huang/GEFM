function [x,r,normR] = OMP_init( A, b, k, opts)
% modified to add initialization, by S.H.

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



% What stopping criteria to use? either a fixed # of iterations,
%   or a desired size of residual:
target_resid    = 1e-6;

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

A_T_nonorth         = zeros(M,k);
A_T_nonorth(:,1:length(indx_set)) = A(:,indx_set);

kk_init = length(indx_set);

indx_set = [indx_set repmat(0, 1, k-length(indx_set))];

x_T=[];

for kk = kk_init+1:k
    
    % -- Step 1: find new index and atom to add
    [dummy,ind_new]     = max(abs(Ar));
    
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
    x_T = A_T_nonorth(:,1:kk)\b;
    %x_T = A_T_nonorth(:,1:kk)'*inv(A_T_nonorth(:,1:kk)*A_T_nonorth(:,1:kk)')*b;
        
    x( indx_set(1:kk) )   = x_T;
    r = b - A_T_nonorth(:,1:kk)*x_T;
    
    normR = norm(r);

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
