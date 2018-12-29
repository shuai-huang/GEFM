function [num] = numerosity(x, threshold)
% numerosity - Finds the numerosity of input matrix x
%        Each column of x is taken to be a vector and
%        any element greater than threshold is assumed
%        to be nonzero.
%
% [num] = numerosity(x)
%
% num       - Integer number of elements with values > threshold (10e-4)
%
% JFM   7/25/2000
% Rev:  6/1/2003

if(~exist('threshold'))
    threshold = 0; %1e-4;
end

[r c] = size(x);
num = zeros(1, c);

for i = 1:c
    n = 0;
    for j = 1:r
        if (abs(x(j, i)) > threshold)
            n = n + 1;
        end
    end
    
    num(1, i) = n;
    
end
