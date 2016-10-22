function method_1(dataset, saveFeatureVector)

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

    geometricFeatures_6 = []; geometricFeatures_10 = []; geometricFeatures_21 = [];
    orientationMap = []; distanceMap = [];
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
        % Enlarge the image
        morphImg = enlargenImage(obj, morphImg, 500, 500, 0);
        % Extract hand
        hand = extractHand(obj, morphImg, 0);
        % Rotate the hand
        rotated = rotateHandImage(obj, hand, 0);
        % Find reference point
        r = findReferencePoint(obj, rotated, 0);  
        % Extract hand contour
        contour = extractHandContour(obj, rotated, 0);
        % Extract peak and valley points
        [peaks, peakInx, valleys, valleyInx, composite, compositeInx, midPoints] = extractFingerPoints(obj, rotated, contour, r, 0);
%         % Cut the wrist of the hand
%         [cut, cutboundary] = cutWrist(obj, rotated, contour, peaks, composite, peakInx, compositeInx, 1);
        % Rotate image again
        rotated2 = rotateHandImage2(obj, rotated, peaks(:,3), midPoints(:,3), 0);       
%         rotated2 = scaleImage(obj, rotated2, 200, 0);
        
        r2 = findReferencePoint(obj, rotated2, 0); 
        contour2 = extractHandContour(obj, rotated2,0);
        % Extract peak and valley points
        [peaks2, peakInx2, valleys2, valleyInx2, composite2, compositeInx2, midPoints2] = extractFingerPoints(obj, rotated2, contour2, r2, 0);
        [registeredContour] = registerFingers(obj, rotated2, contour2, peaks2, valleyInx2, compositeInx2, midPoints2, 0);
        
        % Extract the geometric feature vectors
        [g6, g10, g21] = extractGeometricFeatures(obj, peaks, midPoints);
        % Save the geometric features
        geometricFeatures_6{counter} = g6;
        geometricFeatures_10{counter} = g10;
        geometricFeatures_21{counter} = g21;
        
        % Extract orientation and distance feature vectors
        [orientMap, distMap] = extractDistanceAndOrientationFeatures(obj, registeredContour, r2, 50, 0);
        % Save the orientation and distance feature vectors
        orientationMap(counter,:) = orientMap;
        distanceMap(counter,:) = distMap;
        
    end

    if saveFeatureVector == 1
        if exist('mat_files') ~= 7
            mkdir('mat_files')
        end
        save('./mat_files/geometric6_IITD.mat','geometricFeatures_6')
        save('./mat_files/geometric10_IITD.mat','geometricFeatures_10')
        save('./mat_files/geometric21_IITD.mat','geometricFeatures_21')
        
        save('./mat_files/orientationMap_IITD.mat','orientationMap')
        
        save('./mat_files/distanceMap_IITD.mat','distanceMap')
    end
    
elseif strcmp(dataset,'Bosphorus')
    
    dataset_path = bosphorus_path;
    images = dir(dataset_path);

    discardedImages = [339,340,341, 471,472,473, 225,226,227,264,265,266,429,430,431,579,580,581,591,592,593];
    imageIndices = 3:size(images,1); % skip first 3 files, because they are '.', '..', and '.dstore'
    imageIndices = imageIndices(~ismember(imageIndices, discardedImages));

    geometricFeatures_6 = []; geometricFeatures_10 = []; geometricFeatures_21 = [];
    orientationMap = []; distanceMap = [];
    counter=0;

%     for i=imageIndices
    for i=44;
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
        % Extract hand contour
        contour = extractHandContour(obj, rotated, 0);
        % Extract peak and valley points
        [peaks, peakInx, valleys, valleyInx, composite, compositeInx, midPoints] = extractFingerPoints(obj, rotated, contour, r, 0);
        % Cut the wrist of the hand
        [cut, cutboundary] = cutWrist(obj, rotated, contour, peaks, composite, peakInx, compositeInx, 0);
        rotated2 = rotateHandImage2(obj, cut, peaks(:,3), midPoints(:,3), 0);
        % Find the reference point
        r2 = findReferencePoint(obj, rotated2, 0); 
        contour2 = extractHandContour(obj, rotated2,0);
        [peaks2, peakInx2, valleys2, valleyInx2, composite2, compositeInx2, midPoints2] = extractFingerPoints(obj, rotated2, contour2, r2, 0);
        [registeredContour] = registerFingers(obj, rotated2, contour2, peaks2, valleyInx2, compositeInx2, midPoints2, 0);
        
        % Extract the geometric feature vectors
        [g6, g10, g21] = extractGeometricFeatures(obj, peaks, midPoints);
        % Save the geometric features
        geometricFeatures_6{counter} = g6;
        geometricFeatures_10{counter} = g10;
        geometricFeatures_21{counter} = g21;
        
        % Extract orientation and distance feature vectors
        [orientMap, distMap] = extractDistanceAndOrientationFeatures(obj, registeredContour, r2, 50, 0);
        % Save the orientation and distance feature vectors
        orientationMap(counter,:) = orientMap;
        distanceMap(counter,:) = distMap;
        
    end

    if saveFeatureVector == 1
        if exist('mat_files') ~= 7
            mkdir('mat_files')
        end
        save('./mat_files/geometric6_Bosphorus.mat','geometricFeatures_6')
        save('./mat_files/geometric10_Bosphorus.mat','geometricFeatures_10')
        save('./mat_files/geometric21_Bosphorus.mat','geometricFeatures_21')
        
        save('./mat_files/orientationMap_Bosphorus.mat','orientationMap')
        
        save('./mat_files/distanceMap_Bosphorus.mat','distanceMap')
    end
    
end

end