function w = shannon_ef_sg( x, p, precision)
%computeWeights
x_abs = abs(x);
x = x_abs.^p;
w = [];
xSum = sum(x);

x_ori=x;
x(x_ori==0)=1;
x_log=log(x);
x(x_ori==0)=0;

if ((max(x)+precision)<xSum)
	w = -log(x+precision)/xSum + sum(x.*x_log)/(xSum^2);
    w = p*((x_abs+precision).^(p-1)).*w;
else
	w = repmat(1, size(x,1), size(x,2));
end

