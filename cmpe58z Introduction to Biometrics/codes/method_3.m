function method_3(dataset, saveFeatureVector)

obj = myHandRecognitionClass;

IITD_path = 'C:\Users\Abdulkadir\Desktop\Datasets\IITD';
bosphorus_path = 'C:\Users\Abdulkadir\Desktop\Datasets\bosphorus_hand_db\bosphorus hand db';


if strcmp(dataset,'IITD')
    
    dataset_path = IITD_path;
    images = dir(dataset_path);

    load('mat_files/discardedImageIndices.mat');

    discardedImages = discardedImageIndices;
    imageIndices = 3:size(images,1); % skip first 3 files, because they are '.', '..'
    imageIndices = imageIndices(~ismember(imageIndices, discardedImages));

    contours = [];
    counter=0;
    for i=imageIndices
        fprintf('The processing image id : %d \n', i)
        counter = counter + 1;
        
        imageName = images(i).name;
        imagePath = strcat(dataset_path, '\',imageName);
        image = imread(imagePath);
        % Binarization
        binary = applyBinarization(obj, image, 0.35, 0);
        % Segmentation
        morphImg = applyMorphologicalOps(obj, binary, 1, 0);
        % Enlarge image
        morphImg = enlargenImage(obj, morphImg, 500, 500, 0);
        % Extract hand
        hand = extractHand(obj, morphImg, 0);
        % Rotate image
        rotated = rotateHandImage(obj, hand, 0);
        % Find reference point
        r = findReferencePoint(obj, rotated, 0);   
        % Extract contour
        contour = extractHandContour(obj, rotated, 0);
        % Extract peak and valley points
        [peaks, peakInx, valleys, valleyInx, composite, compositeInx, midPoints] = extractFingerPoints(obj, rotated, contour, r, 0);
        % Cut the wrist of the hand
%         [cut, cutboundary] = cutWrist(obj, rotated, contour, peaks, composite, peakInx, compositeInx, 0);
        rotated2 = rotateHandImage2(obj, rotated, peaks(:,3), midPoints(:,3), 0);
        % Find reference point
        r2 = findReferencePoint(obj, rotated2, 0); 
        % Extract hand contour
        contour2 = extractHandContour(obj, rotated2,0);
        [peaks2, peakInx2, valleys2, valleyInx2, composite2, compositeInx2, midPoints2] = extractFingerPoints(obj, rotated, contour2, r2, 0);
        [registeredContour] = registerFingers(obj, rotated2, contour2, peaks2, valleyInx2, compositeInx2, midPoints2, 0);
        % Shift contour
        shiftedContour = shiftContour(obj, rotated2, registeredContour, midPoints2, 0);
                
        % Save the contours
        contours{counter} = shiftedContour;
    end

    if saveFeatureVector == 1
        if exist('mat_files') ~= 7
            mkdir('mat_files')
        end
        save('./mat_files/contours_IITD.mat','contours')
    end
    
elseif strcmp(dataset,'Bosphorus')
    
    dataset_path = bosphorus_path;
    images = dir(dataset_path);

    discardedImages = [339,340,341, 471,472,473, 225,226,227,264,265,266,429,430,431,579,580,581,591,592,593];
    imageIndices = 3:size(images,1); % skip first 3 files, because they are '.', '..', and '.dstore'
    imageIndices = imageIndices(~ismember(imageIndices, discardedImages));

    contours = [];
    counter=0;
%     for i=imageIndices
    for i=5
        fprintf('The processing image id : %d \n', i)
        counter = counter + 1;
        
        imageName = images(i).name;
        imagePath = strcat(dataset_path, '\',imageName);
        image = imread(imagePath);
        % Binarization
        binary = applyBinarization(obj, image, -1, 0);
        % Segmentation
        morphImg = applyMorphologicalOps(obj, binary, 2, 0);
        % Extract hand
        hand = extractHand(obj, morphImg, 0);
        % Rotate image
        rotated = rotateHandImage(obj, hand, 0);
        % Find reference point
        r = findReferencePoint(obj, rotated, 0);   
        % Extract contour
        contour = extractHandContour(obj, rotated, 0);
        % Extract peak and valley points
        [peaks, peakInx, valleys, valleyInx, composite, compositeInx, midPoints] = extractFingerPoints(obj, rotated, contour, r, 0);
        % Cut the wrist of the hand
        [cut, cutboundary] = cutWrist(obj, rotated, contour, peaks, composite, peakInx, compositeInx, 0);
        rotated2 = rotateHandImage2(obj, cut, peaks(:,3), midPoints(:,3), 0);
        % Find reference point
        r2 = findReferencePoint(obj, rotated2, 0); 
        % Extract hand contour
        contour2 = extractHandContour(obj, rotated2,0);
        % Find peaks and valleys
        [peaks2, peakInx2, valleys2, valleyInx2, composite2, compositeInx2, midPoints2] = extractFingerPoints(obj, rotated2, contour2, r2, 0);
        [registeredContour] = registerFingers(obj, rotated2, contour2, peaks2, valleyInx2, compositeInx2, midPoints2, 0);
        % Shift contour
        shiftedContour = shiftContour(obj, rotated2, registeredContour, midPoints2, 0);
            
        % Save the contours
        contours{counter} = shiftedContour;
    end

    if saveFeatureVector == 1
        if exist('mat_files') ~= 7
            mkdir('mat_files')
        end
        save('./mat_files/contours_Bosphorus.mat','contours')
    end
    
end

end



function method_11(dataset)

if strcmp(dataset,'Bosphorus')
    
    bosphorus_path = 'C:\Users\Abdulkadir\Desktop\Datasets\bosphorus_hand_db\bosphorus hand db';
    save_path = 'C:\Users\Abdulkadir\Desktop\Datasets\bosphorus_hand_db';
    images = dir(bosphorus_path);

    discardedImages = [340,341,342, 472,473,474,      226,227,228,265,266,267,430,431,432,580,581,582,592,593,594];
    imageIndices = 4:size(images,1);
    imageIndices = imageIndices(~ismember(imageIndices, discardedImages));

    geometricFeatures = []; orientationMap=[]; distanceMap=[];
    counter=0;
    for i=imageIndices
    % for i=4:size(images,1) % skip first 3 files, because they are '.', '..', and '.dstore'
%     for i=130
    i
    counter = counter + 1;
    imageName = images(i).name;
    imagePath = strcat(bosphorus_path, '\',imageName);
        
    binary = applyBinarization(imagePath);
%     figure, imshow(binary), title('binary')
    morphImg = applyMorphologicalOps(binary);
%     figure, imshow(morphImg), title('after morph ops')
    hand = extractHand(morphImg);
%     figure, imshow(hand), title('hand extracted') 
    rotated = rotateHandImage(hand);
%     figure, imshow(rotated), title('rotated hand')
    r = findReferencePoint(rotated);   
    contour = extractHandContour(rotated);
    % Extract peak and valley points
    [peaks, peakInx, valleys, valleyInx, composite, compositeInx, midPoints] = extractFeaturePoints(contour, r, 0);
        
    [cut, cutboundary] = cutWrist(rotated, contour, peaks, composite, peakInx, compositeInx);
    figure, imshow(cut), title('Cut hand')
    rotated2 = rotateHandImage2(cut, peaks(:,3), midPoints(:,3));
%     figure, imshow(rotated2), title('Rotated after cutting')
    contour2 = extractHandContour(rotated2);
    [peaks2, peakInx2, valleys2, valleyInx2, composite2, compositeInx2, midPoints2] = extractFeaturePoints(contour2, r, 0);
%     figure, imshow(rotated2), hold on, plot(peaks2(1,:),peaks2(2,:),'rx')
    
    [newContour] = registerFingers2(rotated2, contour2, peaks2, valleyInx2, compositeInx2, midPoints2);
    
%     shiftedContour = shiftContour(rotated2, newContour, midPoints2);
%     contours{counter} = shiftedContour;

%     [g_21] = extractGeometricFeatures(peaks2, midPoints2, 0);
%     geometricFeatures(counter,:) = g_21;

    r2 = findReferencePoint(rotated2);  
    [orientMap, distMap] = extractDistanceAndOrientationFeatures(newContour, r2);
    
    orientationMap(counter,:) = orientMap;
    distanceMap(counter,:) = distMap;

    
    
%   saveFigures(strcat(save_path, '\outputs2\',num2str(i-3), '.bmp'), cut, peaks, valleys, composite, midPoints);

    end
    
end
    save('orientationMap.mat', 'orientationMap');
    save('distanceMap.mat', 'distanceMap');
    
%     save('geometricFeatures.mat', 'geometricFeatures')
%     save('contours.mat','contours')

end