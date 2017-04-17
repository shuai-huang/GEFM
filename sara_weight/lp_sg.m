function y = lp_sg(x,p,precision)
% supergradient of lp penalty

x = abs(x) ;
y = p*(x+precision).^(p-1) ; % 
