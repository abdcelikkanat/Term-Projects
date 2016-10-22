function main()
 
    file = load('data-seq.mat');
    data = file.data;
    numOfSeqs = length(data);

	data = symbol2numeric(data,'abcd');
   
    N = 15; L = 4;

    [F,predict] = vbhmm(data,N,L,100,0.0001);
 
    figure
    grid;
    plot(1:length(F),F,'-')
    
    figure,
    grid;
    plot(diff(F),'-');
    xlabel('Iterations'), ylabel('\Delta F')
    title('\Delta')
    
 %  hinton(predict.WA), hinton(predict.WB'), hinton(predict.WPi')


end
  
 
function [F, predict] = vbhmm(data, N, L, maxIter, epsilon)
% Input parameters:
%   N -> number of states
%   L -> number of observations    

    numOfSeqs = size(data,2);

    totalSeqLength = 0;
    for i=1:numOfSeqs
        totalSeqLength = totalSeqLength + size(data{i},2);
    end

    % Initialise the pseudo-counts
    uA = ones(1,N)*(1/N);
    uB = ones(1,L)*(1/L);
    uPi = ones(1,N)*(1/N);
    % Pick an HMM from the prior to initialize the counts
    sum_wa = zeros(N,N); sum_wb = zeros(N,L);
    for n=1:N, % loop over hidden states
        sum_wa(n,:) = (dirichlet(uA',1))*totalSeqLength;
        sum_wb(n,:) = (dirichlet(uB',1))*totalSeqLength;
    end;
    w_pi = (dirichlet(uPi',1))*numOfSeqs;
    
    A = zeros(N,N); B = zeros(N,L); Pi = zeros(N,1);
    for i=1:maxIter
        
        WA = sum_wa + repmat(uA,N,1);
        WB = sum_wb + repmat(uB,N,1);
        WPi = w_pi + uPi;
    
        A = exp(  psi(WA) - repmat( psi(sum(WA,2)) ,[1 N])  );
        B = exp(  psi(WB) - repmat( psi(sum(WB,2)) ,[1 L])  );
        Pi = exp(  psi(WPi) - psi(sum(WPi,2))  );  
     

        % E Step
        [alpha, beta, gamma, xsi, gammak, a_estt, gammaInit, logZ] = forwardBackward(A', B, Pi', data);
        sum_wa = a_estt;
        sum_wb = gammak;
        w_pi = gammaInit; 
       % [sum_wA, sum_wB, w_Pi, logZ(i), lnZv] = forwback(A,B,Pi,data)

        % Compute F, straight after E Step.
        Fa(i)=0; Fb(i)=0; Fpi(i)=0;
        for kk = 1:N,
            Fa(i) = Fa(i) - KL_Dirichlet(WA(kk,:),uA);
            Fb(i) = Fb(i) - KL_Dirichlet(WB(kk,:),uB);
        end;
        Fpi(i) = - KL_Dirichlet(WPi,uPi);
        
        F(i) = logZ + Fa(i)+Fb(i)+Fpi(i);
  
        if i>2
            %if( (F(i) - F(i-1)) < epsilon )
            if( (F(i) - F(2)) < (1+epsilon)*(F(i-1)-F(2)))
                fprintf('Converged at %d th iteration.\n',i);
                break;
            end
        end
    end
    
    predict = struct('A', A, 'B', B, 'Pi', Pi,'WA',WA,'WB',WB,'WPi',WPi);

end

function [result] = KL_Dirichlet(alpha, beta)
% Input parameters :
% alpha(1xN), and beta(1xN) are paramaeters of Dirichlet distribution

result = gammaln(sum(alpha))-gammaln(sum(beta)) - sum(gammaln(alpha)-gammaln(beta)) ...
          + (alpha-beta) * (psi(alpha)-psi(sum(alpha)))' ;
 
end


% Generates samples from Drichlet distribution
function [x] = dirichlet(alpha, N)
% Paramaters :
%     alpha -> Nx1 matrix,

    L = size(alpha,1);
    x = gamrnd(repmat(alpha',N,1),1,N,L);
    x = bsxfun(@rdivide, x, sum(x,2));
    
end  

