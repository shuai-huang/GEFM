function U=denoise_asbs(B,lambda,Psi,Psit,W,Q,pars)

    % The Alternating Split Bregman Shrinkage (ASBS) algorithm
    % 
    % S. Setzer, Split Bregman algorithm, Douglas-Rachford splitting and frame shrinkage, pp. 464–476, Springer Berlin Heidelberg, Berlin, Heidelberg, June 2009.
    %
    % By Shuai Huang, The Johns Hopkins University
    % Email: shuang40@jhu.edu 
    % Date: 09/16/2016
    %


    % Assigning parameres according to pars and/or default values
    denoiseiter=pars.denoiseiter;
    innertol=pars.innertol;
    gamma=pars.gamma;

    [m,n]=size(B);
    F=zeros(size(W));
    U=B;
    i=0;

    while(i<denoiseiter)
        Psi_U_tmp = Psi(U);
        D=F+Psi_U_tmp;
        D=max(abs(D)-gamma*lambda*W, 0).*sign(D);
        F=F+Psi_U_tmp-D;
        U_pre=U;
        U=Q*(gamma*B+Psit(D-F));
        cri = norm(U_pre-U, 'fro')/norm(U,'fro');
        %fprintf('%5.8f\n', cri)
        if (cri<=innertol)
            break;
        end
    end

end
