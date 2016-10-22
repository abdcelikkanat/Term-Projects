function analyze_similarity_matrix(SM, Class_Labels, nbin, interval)

% Get the total size
N = numel(Class_Labels);

% inx1 and inx2 are usead as indexes
inx1 = 1; inx2 = 1;
%% Store the genuine and imposter scores
for i=1:N
    for j=i+1:N
        if(Class_Labels(i) == Class_Labels(j))
            genuine(inx1) = SM(i,j);
            genuine_scores(inx1) = SM(i,j);
            inx1 = inx1 + 1;
        else
            imposter(inx2) = SM(i,j);
            imposter_scores(inx2) = SM(i,j);
            inx2 = inx2 + 1;
        end
    end
end

%% Find PDFs
% Extract the pdf of genuine
[counts, genuine_centers] = hist(genuine_scores, nbin);
genuine_pdf = counts/trapz(genuine_centers, counts);

% Extract the pdf of imposter
[counts, imposter_centers] = hist(imposter_scores, nbin);
imposter_pdf = counts/trapz(imposter_centers, counts);

% Plot the pdf of genuine and imposter
figure
plot(genuine_centers, genuine_pdf,'-b',imposter_centers, imposter_pdf,'-r')
xlabel('Score'), ylabel('Density')
legend('Genuine','Imposter')
title('Pdfs of Genuine and Imposter Scores')

%Find the min. and max. of data to determine data limits
minVal = min(SM(:));
maxVal = max(SM(:));

countLimit = round((maxVal-minVal)/interval)+1;

% Intialize t, FAR, and GMR vectors
t = zeros(1,countLimit); 
FAR = zeros(1,countLimit); 
GMR = zeros(1,countLimit);

%% Find the FAR and GAR
t(1) = minVal;
for i=1:countLimit
   FAR(i) = sum(imposter_scores<t(i));
   GMR(i) = sum(genuine_scores<t(i));
   t(i+1) = t(i) + interval;
end
FAR = FAR / numel(imposter_scores);
GMR = GMR / numel(genuine_scores);

% Now, Calculate the FRR
FRR = 1 - GMR;

% Plot the error rates graph
figure
plot(t(1:countLimit),FAR,'r',t(1:countLimit),FRR,'b')
legend('FAR','FRR')
ylabel('Error rates')
title('Error Rates')


%% Find the Equal Error Rate with EER threshold %%

% Find the first closest point to the intersection
[minvalue, minInx1] = min(abs(FAR-FRR));
% Find the second closest point to the intersection
if FAR(minInx1)>FRR(minInx1)
    minInx2 = minInx1+1;
else
    minInx2 = minInx1-1;
end

% Calculate the EER threshold
EER_threshold = (t(minInx1) + t(minInx2))/2;
% Calculate the EER
EER = (FAR(minInx1) + FAR(minInx2) + FRR(minInx1) + FRR(minInx2))/4;
% Print the EER results
fprintf('EER : %f%% \n',EER*100)
fprintf('EER threshold: %f \n', EER_threshold)

%% Plot the ROC curve
roc_in_log_domain = 0;
if roc_in_log_domain == 1
    figure
    subplot(1,2,1)
    plot(FAR, GMR, 'r-')
    title('ROC Curve')
    subplot(1,2,2)
    plot(log(FAR),log(GMR),'r-')
    legend('ROC Curve in Log domain')
else
    figure,
    plot(FAR, GMR, 'r-')
    title('ROC Curve')
    xlabel('False Positive Rate')
    ylabel('True Positive Rate')
end

%% Find the FRR values at different FAR values

% Scale the percentage values to the interval [0,1]
far_vals = [0.1, 1, 10]/100;
for i=1:3
    % Find the indexes of closest points to the FAR vector
    [minvalue, minInx1] = min(abs(FAR - far_vals(i)));
    % Print the values
    fprintf('FRR : %.2f%% at FAR point : %.1f%% \n',FRR(minInx1)*100,far_vals(i)*100)
end

end