function classification( )

dataset = 'bosphorus';
method = 31;

if strcmp(dataset,'bosphorus')

% Read the similarity matrix of feature vectors
if method == 11
    s = load('mat_files/bosphorus_g21_similarity.mat');
    s = s.g21_similarity;
elseif method == 12
    s = load('mat_files/bosphorus_distanceMap_similarity.mat');
    s = s.distanceMap_similarity; 
elseif method == 13
    s = load('mat_files/bosphorus_orientationMap_similarity.mat');
    s = s.orientationMap_similarity;         
elseif method == 31
    s = load('mat_files/contour_bosphorus_similarity.mat');
    s = s.contour_similarity;       
end
% Convert upper diagnoal matrix to symetric matrix
S = s + triu(s,1)';

% Each person contains only 3 images
M = size(S,1); % M is the number of images
N = M/3; % N is the number of people
% person is a Nx3 matrix containing indices of persons' image similarities
person = reshape(1:M, 3, N)'; 
% Permute the image indices to randomly choose images
for id=1:N
    person(id,:) = person(id, randperm(3));
end

% Set population size and enrollement set size
enrollSize = 2;
populationSize = 20;


if enrollSize == 2
    testSetInx = [3; 2; 1];
    enrollmentSetInx = [1, 2; 1, 3; 2, 3;];    
    numOfExperiments = 3;
elseif enrollSize == 1
    testSetInx = [1; 1; 2; 2; 3; 3];
    enrollmentSetInx = [2; 3; 1; 3; 1; 2];
    numOfExperiments = 6;
end


experimentResult = zeros(numOfExperiments,1);
for ex=1:numOfExperiments
    personId = 1:populationSize;

    testInx = person(:,testSetInx(ex));
    enrollmentInx = person(:,enrollmentSetInx(ex,:));

    trueClass = 0;
    for id=personId
        [val, inx] = sort(S(testInx(id),:));
        pred = inx(2); % Choose the second closest, since the first one is itself
        trueClass = trueClass + ismember(pred, enrollmentInx(id,:));

    end

    experimentResult(ex) = (trueClass / populationSize) * 100;

end
m = mean(experimentResult);
s = std(experimentResult);
fprintf('Population size : %d Mean : %f Std : %f \n', populationSize, m, s);

end



end

