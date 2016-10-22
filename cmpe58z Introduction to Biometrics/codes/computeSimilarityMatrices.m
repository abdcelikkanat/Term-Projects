function computeSimilarityMatrices()

var = 113 % Change this parameter depending on your usage

    % For geometric features of Bosphorus dataset 
    if var == 113

        gf = load('./mat_files/geometric21_Bosphorus.mat');
        computeGeometricSimilarity(gf.geometricFeatures_21, 21, 'chi2', 'bosphorus')

        s = load('./mat_files/bosphorus_g21_similarity.mat');

        classLabels = [];
        for i=1:size(s.g21_similarity,2)
            class = floor((i-1)/3)+1;
            classLabels = [classLabels, class];
        end
        
        analyze_similarity_matrix(s.g21_similarity, classLabels, 200, 0.001);
    % For distance features of Bosphorus dataset 
    elseif var == 121

        gf = load('./mat_files/distanceMap_Bosphorus.mat');
        computeDistanceMapSimilarity(gf.distanceMap, 'l2','bosphorus')

        s = load('./mat_files/bosphorus_distanceMap_similarity.mat');

        classLabels = [];
        for i=1:size(s.distanceMap_similarity,2)
            class = floor((i-1)/3)+1;
            classLabels = [classLabels, class];
        end
        
        analyze_similarity_matrix(s.distanceMap_similarity, classLabels, 100, 0.1);        
    
    % For orientation features of Bosphorus dataset         
    elseif var == 131

        gf = load('./mat_files/orientationMap_Bosphorus.mat');
        computeOrientationMapSimilarity(gf.orientationMap, 'l2', 'bosphorus')

        s = load('./mat_files/bosphorus_orientationMap_similarity.mat');

        classLabels = [];
        for i=1:size(s.orientationMap_similarity,2)
            class = floor((i-1)/3)+1;
            classLabels = [classLabels, class];
        end
        
        analyze_similarity_matrix(s.orientationMap_similarity, classLabels, 100, 0.1);         
  
    % For contour features of Bosphorus dataset    
    elseif var == 141

        gf = load('./mat_files/contours_Bosphorus.mat');
        computeContourSimilarity(gf.contours, 'bosphorus')

        s = load('./mat_files/contour_bosphorus_similarity.mat');

        classLabels = [];
        for i=1:size(s.contour_similarity, 2)
            class = floor((i-1)/3)+1;
            classLabels = [classLabels, class];
        end
        
        analyze_similarity_matrix(s.contour_similarity, classLabels, 100, 0.1);          
        
    elseif var == 213
        %Load class labels
        l = load('./mat_files/IITD_classLabels.mat');
        
        gf = load('./mat_files/geometric21_IITD.mat');
        computeGeometricSimilarity(gf.geometricFeatures_21, 21, 'chi2', 'IITD')

        s = load('./mat_files/IITD_g21_similarity.mat');

        classLabels = l.classLabels';
        
        analyze_similarity_matrix(s.g21_similarity, classLabels, 100, 0.001);        
        
    elseif var == 221
        %Load class labels
        l = load('./mat_files/IITD_classLabels.mat');
        
        gf = load('./mat_files/distanceMap_IITD.mat');
        computeDistanceMapSimilarity(gf.distanceMap, 'l2', 'IITD')

        s = load('./mat_files/IITD_distanceMap_similarity.mat');

        classLabels = l.classLabels';

        analyze_similarity_matrix(s.distanceMap_similarity, classLabels, 100, 1);  
   
    elseif var == 231
        %Load class labels
        l = load('./mat_files/IITD_classLabels.mat');
        
        gf = load('./mat_files/orientationMap_IITD.mat');
        computeOrientationMapSimilarity(gf.orientationMap, 'l2', 'IITD')

        s = load('./mat_files/IITD_orientationMap_similarity.mat');

        classLabels = l.classLabels';
        
        analyze_similarity_matrix(s.orientationMap_similarity, classLabels, 100, 0.01);  

    elseif var == 241
        %Load class labels
        l = load('./mat_files/IITD_classLabels.mat');
        
        gf = load('./mat_files/contours_IITD.mat');
        computeContourSimilarity(gf.contours, 'IITD')

        s = load('./mat_files/contour_IITD_similarity.mat');

        classLabels = l.classLabels';
        
        analyze_similarity_matrix(s.contour_similarity, classLabels, 100, 0.1);           
    
    elseif var == 0
        
        
        
    else
        
        doFuse()
        
    end
    
end


function doFuse()
dataset = 'Bosphorus';

o = load('./mat_files/Bosphorus_orientationMap_similarity.mat');
N = size(o.orientationMap_similarity,2);
o = normalize(o.orientationMap_similarity, 'max_min');


d = load('./mat_files/Bosphorus_distanceMap_similarity.mat');
d = normalize(d.distanceMap_similarity, 'max_min');

g = load('./mat_files/Bosphorus_g21_similarity.mat');
g = normalize(g.g21_similarity, 'max_min');

if strcmp(dataset,'IITD')
    l = load('./mat_files/IITD_classLabels.mat');
    classLabels = l.classLabels';
end
if strcmp(dataset,'Bosphorus')
    classLabels = [];
     for i=1:N
        class = floor((i-1)/3)+1;
        classLabels = [classLabels, class];
    end
end

sim = (o + d + g);
analyze_similarity_matrix(sim, classLabels, 200, 0.001)

end

function [output] = normalize(similarity, method)
    if strcmp(method, 'max_min')
        M = triu(similarity) + triu(similarity,1)';
        mx = max(max(M)); mn = min(min(M));
        output = triu((M-mn)/(mx-mn));
    end

    
end

function [contour_similarity] = computeContourSimilarity(contours, name)

    N = numel(contours);
    contour_similarity = zeros(N, N);
    for i=1:numel(contours)
        fprintf('The processing image id : %d \n', i)
        
        for j=i:numel(contours)    
            contour_similarity(i,j) =  computeDist('hausdorff', contours{i}, contours{j});
        end

    end
    
    save(strcat('mat_files/contour_',name,'_similarity.mat'),'contour_similarity')

end

function computeGeometricSimilarity(dist, w, metric, name)

    N = size(dist,2);
    g21_similarity = zeros(N,N); 
    g10_similarity = zeros(N,N);
    g6_similarity = zeros(N,N);
    for i=1:N
       for j=i:N
            if w == 21
                g21_similarity(i,j) = computeDist(metric, dist{i}, dist{j});
            elseif w == 10
                g10_similarity(i,j) = computeDist(metric, dist{i}, dist{j});
            elseif w == 6
                g6_similarity(i,j) = computeDist(metric, dist{i}, dist{j});
            end
       end
    end

    if w == 21
        save(strcat('mat_files/',name,'_g21_similarity.mat'),'g21_similarity');
    end
    if w == 10
        save(strcat('mat_files/',name,'_g10_similarity.mat'),'g10_similarity');
    end
    if w == 6
        save(strcat('mat_files/',name,'_g6_similarity.mat'),'g6_similarity');
    end

end


function computeDistanceMapSimilarity(featureVec, metric, name)

    N = size(featureVec,1);
    distanceMap_similarity = zeros(N,N);
    for i=1:N
       for j=i:N
            distanceMap_similarity(i,j) = computeDist(metric, featureVec(i,:), featureVec(j,:));
       end
    end
    
    save(strcat('./mat_files/',name,'_distanceMap_similarity.mat'),'distanceMap_similarity');

end



function computeOrientationMapSimilarity(featureVec, metric, name)

    N = size(featureVec,1);
    orientationMap_similarity = zeros(N,N);
    for i=1:N
       for j=i:N
            orientationMap_similarity(i,j) = computeDist(metric, featureVec(i,:), featureVec(j,:));
       end
    end
    
    save(strcat('./mat_files/',name,'_orientationMap_similarity.mat'),'orientationMap_similarity');

end

function computeSimilarityMatrix(featureVec, metric)

    N = size(featureVec,1);
    orient_similarity = zeros(N,N);
    for i=1:N
       for j=i:N
            orient_similarity(i,j) = pdist2(featureVec(i,:), featureVec(j,:), metric );
       end
    end
    
    save('orient_similarity.mat','orient_similarity');

end