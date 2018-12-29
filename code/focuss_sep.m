function [ x ] = focuss_sep( y, A_one, A_two, targetdiv, positive_code, xinit, xiter, p, lambdamax, mu)  
%
% This is the modified regularized focuss algorithm with additional initialization xinit. By S.H.
%
%
%  [ x ] = focuss( y, A, targetdiv, positive_code, xiter, p, lambdamax )
%
%       Function to find the solution to Ax = y where x and
%       y are matricies (of column vectors).  Each solution vector x 
%       is sparse with the same non-zero indices. 
%
%       See notes 9/2/2003 and Adler:1996
%
%       Inputs  y - input matrix of solutions
%               A - dictionary matrix
%               targetdiv - required sparsity (JFM: maximum sparsity allowed) 
%                           (-1 to allow normal convergence, no sparsifying)
%               positive_code - True if the x coefficients are restricted to >= 0
%               xiter - Number of iterations
%               p - diversity measure
%               lambdamax - Highest value of the regularization parameter
%
%       Outputs NumNonzeroCoeffs - the number of nonzero coeffs in
%                   each column of the solution matrix
%               indices - the indices selected
%               x - Output vector (solution)
%
%  Copyright 2005 Joseph F. Murray
%
% JFM:  10/20/2003
% Rev:  3/10/2004

A = [A_one A_two];

[m N] = size(y);
[m n] = size(A);

%x = pinv(A) * y;
%x = rand(n,N); % Uniform distribution on [0,1]
x = xinit;  % modified here
%x = zeros(n,N);

% Iterate with FOCUSS (-CNDL version) on each pattern until convergence

% Focuss parameters
%lambdamax = 2.0e-3;  % Regularization parameter limit
%xiter = 15;          % FOCUSS iterations
%p = 0.5;             % Norm

%figure(1);

for k = 1:N
    if(norm(y(:,k)) == 0)
        x(:,k) = zeros(n, 1);
        continue;
    end

    for index_xiter = 1:xiter
        x_pre = x(:,k);
        err = y(:,k) - A*x(:,k) ;        % Error vector
        
        resid = norm( err );       
        num = numerosity(x(:,k),1e-2);
        x_norm = norm(x(:,k),1);
        y_norm = norm(y(:,k));
        
        % Stopping condition
        %  if(resid < params.focussconverge)
      %      break;
      %  end
      
        if(y_norm == 0) 
            epsilon = 0;
        else 
            epsilon = resid / y_norm;
        end
        %lambdax = (1-min(epsilon,1)) * lambdamax;
        lambdax = lambdamax;
        
        % -- Display --
%         fprintf('  k = %d  iter = %d  resid = %f  num = %d  x_norm = %f', k, index_xiter, resid, num, x_norm);       
%         
%         subplot(3,1,1); plot(x(:,k),'.'); title('x_k');
%         subplot(3,1,2); plot(y(:,k),'.'); title('y_k');
%         subplot(3,1,3); plot(err,'.'); title('err_k');
%         
%         pause(.05);
        % -- End Display --
         
        pinvPi_seq = abs(x(:,k)).^(2-p);
        pinvPi_seq((size(A_one,2)+1):end) = 1/mu*pinvPi_seq((size(A_one,2)+1):end);
        %pinvPi = diag( abs(x(:,k)).^(2-p) );
        %pinvPi_err = lambdax * eye(m) + 1e-10 * eye(m);  % Pi inverse error, hardcoded for p = 2

        A_pro = A;
        for (i =1:size(A_pro,2))
            A_pro(:,i) = A_pro(:,i) * pinvPi_seq(i);
        end
        A_pro = A_pro*A';
        for (i =1:size(A_pro,2))
            A_pro(i,i) = A_pro(i,i)+lambdax;
        end
        lambda = A_pro \ y(:,k);
                    
        %lambda = (A * pinvPi * A' + pinvPi_err) \ y(:,k);
        %x(:,k) = pinvPi * A' * lambda;
        x(:,k) = pinvPi_seq .* (A'*lambda);
                
        % Make x positive and restrict values 
        if(positive_code == true) 
            for i = 1:n
                if(x(i,k) < 0) 
                    x(i,k) = 0.0; 
                    %x(i,k) = 0.001;   % This doesn't work well at all
                end
            end
        end        
       
        % Sparsify in the last half of the iterations
        if(index_xiter > xiter/2  &  num > targetdiv  &  targetdiv > 0)
            x(:,k) = pickhighest(x(:,k), targetdiv)';
        end

        if (norm(x_pre - x(:,k), 'fro')/norm(x(:,k), 'fro')<1e-6) 
            break;
        end
        
    end    
 %   disp(sprintf('k = %3d  iter = %d  resid = %f  num = %3d  x_norm_1 = %8.3f max(xi) = %7.3f min(xi) = %7.3f', ...
 %       k, index_xiter, resid, num, x_norm, max(x(:,k)), min(x(:,k))  ));        
end

% % Calculate MSE
% resid = A*x - y;
% clear('err');
% num = 0;
% x_norm = 0;
% 
% for k = 1:N
%     err(k)=norm(resid(:,k))^2;
%     num = num + numerosity(x(:,k),1e-2);
%     x_norm = x_norm + norm(x(:,k),1);
%        
% end
% 
% mse(iter)=sum(err)/(m*N);
% rmse(iter)= sqrt(mse(iter))/ysigma;
% num = num / N;
% x_norm = x_norm / N;
% 
% disp(sprintf('End:  rmse = %f  avg num = %f   avg x_norm_1 = %f', rmse(iter), num, x_norm));
