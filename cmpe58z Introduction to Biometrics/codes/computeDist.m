function [ result ] = computeDist( method_name, param1, param2, param3 )

if nargin == 0
   error('Enter the method name!') 
end

%% Computes the modified Hausdorff distance
if strcmp(method_name, 'hausdorff')
    
    if nargin <= 2
        error('Enter the contours!') 
    end
    contour1 = param1; contour2 = param2;
    
    if nargin == 4
        sample_size = param3;
    else
        sample_size = 2048;
    end
    
    [x1, inx1] = datasample(contour1(:,2),sample_size);
    y1 = contour1(inx1,1);
    [x2, inx2] = datasample(contour2(:,2),sample_size);
    y2 = contour2(inx2,1);
        
    diffX = bsxfun(@minus, repmat(x2',numel(x1),1),x1);
	diffY = bsxfun(@minus, repmat(y2',numel(y1),1),y1);
	dist = sqrt(diffX.^2 + diffY.^2);
            
%     h = max(min(dist,[],2));
%     Modified version
    h1 = sum(min(dist,[],2)) / numel(x1);
    h2 = sum(min(dist,[],1)) / numel(x2);
    result = max(h1, h2);

elseif  strcmp(method_name, 'chi2')   
    if nargin == 3
        result = distChiSq(param1, param2);
    else
        error('Missing parameters!') 
    end
    
 elseif strcmp(method_name, 'cosine')
    
    if nargin == 3
        [h, w] = size(param1);
        if h == 1
            param1 = param1';
        end
        [h, w] = size(param2);
        if h == 1
            param2 = param2';
        end
        
        p1 = sqrt(sum(param1.^2));
        p2 = sqrt(sum(param2.^2));
        
        sim = param1' * param2 / (p1*p2);
        
        % If all vectors are positive
        if sum(param1 < 0) == 0 & sum(param2 < 0) == 0
            d = 2* acos(sim)/pi;
        else
            d = acos(sim)/pi;
        end
        % I convert the range to irs negative space for ROC curves
        result = (1-d);
    else
        error('Missing parameters!') 
    end   
    
elseif strcmp(method_name, 'l2')
    
    if nargin == 3
        result = l2(param1,param2);
    else
        error('Missing parameters!') 
    end
    
elseif strcmp(method_name, 'l1')
    
    if nargin == 3
        result = norm(param1-param2,1);
    else
        error('Missing parameters!') 
    end
    
else
    
    error('Unknown method name!')
    
end

end


function result = l2(x1, x2)

    result = sqrt(sum((x1 - x2).^2));

end

% This function taken from the adress below:
% http://www.cs.columbia.edu/~mmerler/project/code/pdist2.m
function D = distChiSq( X, Y )
%%% supposedly it's possible to implement this without a loop!
m = size(X,1);  n = size(Y,1);
mOnes = ones(1,m); D = zeros(m,n);
for i=1:n
  yi = Y(i,:);  yiRep = yi( mOnes, : );
  s = yiRep + X;    d = yiRep - X;
  D(:,i) = sum( d.^2 ./ (s+eps), 2 );
end
D = D/2;

end
