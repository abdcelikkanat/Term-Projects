% This file contains functions to apply data augmentation for the dataset
function augmentation()
    raw_dataset_path = 'C:\Users\Abdulkadir\Desktop\Datasets\Pet Dataset\raw';
    dataset_path = 'C:\Users\Abdulkadir\Desktop\Datasets\Pet Dataset\dataset';
    %Since all images in the same folder, seperate images of same class
    %from others by putting them into a different folder
    %generateClasses(raw_dataset_path, dataset_path)
    equal_path = 'C:\Users\Abdulkadir\Desktop\Datasets\Pet Dataset\equal';
    % Equalize the images by applying mirroring
    %equalizeImageSizes(dataset_path, equal_path, 128)
    fold_path = 'C:\Users\Abdulkadir\Desktop\Datasets\Pet Dataset\folds';
    % Equally divide the images into subsets to use them for k-fold
    %divideIntoSubsets(equal_path, fold_path, 10)
    augmented_path = 'C:\Users\Abdulkadir\Desktop\Datasets\Pet Dataset\augmented';
    copyAllImages(fold_path, augmented_path)
    
    horizontalFlip(fold_path, augmented_path, 0.5)
    
    rotateImages(fold_path, augmented_path, [-12, -6, 0, 6, 12])
    
    translate(fold_path, augmented_path, [-16, -8, 8, 16])
    
    zoomIn(fold_path, augmented_path, [1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    
    final_path = 'C:\Users\Abdulkadir\Desktop\Datasets\Pet Dataset\final';
    resizeAll(augmented_path, final_path, [64,64])

   
    
    
end

% All images in the same folder, so puts images into 
% their class folder wrt their file names
function generateClasses(base_dir, target_dir)

    imgs = dir(base_dir);
    numofimages = size(imgs,1); % Get the number of images in the folder
    
    for i=3:numofimages
        i
        % Get the imagename
        imagename = strsplit(imgs(i).name,'_');
        % Get the class name
        className = strjoin(imagename(1:end-1),'_');
        image_lastpart = strsplit(char(imagename(end)),'.');
        % Get the image id in the same class
        imageId = image_lastpart(1);
        
        % Read images
        image_path = strcat(base_dir,'\', imgs(i).name);
        image = imread(char(image_path));
        % Get the image size
        [h, w, d] = size(image);
        % If image is grayscale, convert it to rgb image
        if d == 1
            image = image(:,:,[1 1 1]);
        end               
        % Copy the images belonging to the same class into the same folder
        image_save_path = strcat(target_dir,'\', className,'\', imageId, '.jpg');
        % Create a folder for a class if not exists
        if isdir(char(strcat(target_dir,'\',className))) ~= 1
            mkdir(char(strcat(target_dir,'\',className)))
        end
        % Now, save the images
        imwrite(image,char(image_save_path));
        
    end
end

% Mirror the images to equalize their sizes, which is maxSideLength
function equalizeImageSizes(base_dir, target_dir, maxSideLength)

    classFolders = dir(base_dir);
    numofClasses = size(classFolders,1); % Get the total number of classes
    
    % Print the number of classes, -2 caused by '.' and '..'
    fprintf('Total number of classes : %d \n', numofClasses-2);
    
    for i=3:numofClasses % Start from 3, because the first ones are '.' and '..'
        className = classFolders(i).name;
        class_dir = strcat(base_dir,'\',className);
        
        images = dir(class_dir);
        numOfImages = size(images,1); % Get the number of images in the class
        fprintf('Class name %s, and containing %d images \n',className, numOfImages-2)
        for j=3:numOfImages
            % Get the imagename
            imagename = images(j).name;
            % Read images
            image_dir = strcat(class_dir,'\', imagename);
            image = imread(char(image_dir));
            % Get the image size
            [h, w, d] = size(image);  

            % Resize images such that max. of largest side will be 'maxSideLength'
            % Then, mirror the images to equalize the side lengths
            writing = 1;
            if max(h,w) == h
                scaledImg = imresize(image,[maxSideLength,NaN]);
                flipped = flip(scaledImg ,2);
                if maxSideLength < 3*size(scaledImg,2)
                    if maxSideLength >= 2*size(scaledImg,2)
                        mirrored = [flipped(:,1+3*size(scaledImg,2)-maxSideLength:end,:), scaledImg, flipped(:,1:size(scaledImg,2),:)];
                    else
                        mirrored = [scaledImg, flipped(:,1:maxSideLength-size(scaledImg,2),:)];
                    end
                else
                    writing=0;
                end
            else
                scaledImg = imresize(image,[NaN, maxSideLength]);
                flipped = flip(scaledImg ,1);

                if maxSideLength < 3*size(scaledImg,1)
                    if maxSideLength >= 2*size(scaledImg,1)
                        mirrored = [flipped(1+3*size(scaledImg,1)-maxSideLength:end,:,:); scaledImg; flipped(1:size(scaledImg,1),:,:)];
                    else
                        mirrored = [scaledImg; flipped(1:maxSideLength-size(flipped,1),:,:)];
                    end
                else
                    writing=0;
                end
            end
                
            % Copy the images belonging to the same class into the same folder
            image_save_path = strcat(target_dir,'\', className,'\', imagename);
            % Create a folder for a class if not exists
            if isdir(char(strcat(target_dir,'\',className))) ~= 1
                mkdir(char(strcat(target_dir,'\',className)))
            end
            % Now, save the images
            if writing == 1
                imwrite(mirrored,char(image_save_path));
            end
        end
    end
end


function divideIntoSubsets(base_dir, target_dir, numOfSubsets)
    classFolders = dir(base_dir);
    numofClasses = size(classFolders,1); % Get the total number of classes
    
    % Print the number of classes, -2 caused by '.' and '..'
    fprintf('Total number of classes : %d \n', numofClasses-2);
    
    for i=3:numofClasses % Start from 3, because the first ones are '.' and '..'
        className = classFolders(i).name;
        class_dir = strcat(base_dir,'\',className);
        
        images = dir(class_dir);
        numOfImages = size(images,1); % Get the number of images in the class
        fprintf('Class name %s, and containing %d images \n',className, numOfImages-2)
        for k=0:numOfSubsets-1
            foldname = strcat('subset',num2str(k+1));
            % Check a folder for a subset exists or not
            if isdir(char(strcat(target_dir,'\',foldname))) ~= 1
                mkdir(char(strcat(target_dir,'\',foldname)))
            end
            
            if k ~= numOfSubsets-1
                upTo = ceil((numOfImages-3)/numOfSubsets)*(k+1)-1;
                from = ceil((numOfImages-3)/numOfSubsets)*k;
            else
                upTo = numOfImages-3;
                from = ceil((numOfImages-3)/numOfSubsets)*k;
            end
            for j=from+3:upTo+3
                % Get the imagename
                imagename = images(j).name;
                % Read images
                image_dir = strcat(class_dir,'\',imagename);
                image = imread(char(image_dir));
                % Now, copy images
                target_image_dir = strcat(target_dir,'\', foldname,'\',className,'\',imagename);
                % Create a folder for a class if not exists
                if isdir(char(strcat(target_dir,'\', foldname,'\',className))) ~= 1
                    mkdir(char(strcat(target_dir,'\', foldname,'\',className)))
                end
                imwrite(image,char(target_image_dir));
            end
        end
        
    end
end


%% Copy all images from one folder into another
function copyAllImages(base_dir, target_dir)
    subsets = dir(base_dir);
    numOfSubsets = size(subsets,1);

    for k=3:numOfSubsets
        subset_path = strcat(base_dir,'\',subsets(k).name);
        classes = dir(subset_path);
        numOfClasses = size(classes,1);

        target_subset = strcat(target_dir,'\', subsets(k).name);
        if isdir(char(target_subset)) ~= 1
            mkdir(char(target_subset))
        end
        for i=3:numOfClasses

            class_path = strcat(subset_path,'\', classes(i).name);
            images = dir(class_path);

            target_class = strcat(target_subset,'\', classes(i).name);
            if isdir(char(target_class)) ~= 1
                mkdir(char(target_class))
            end
            for j=3:size(images,1);

                image_path = strcat(class_path,'\', images(j).name);
                image = imread(image_path);

                image_save_path = strcat(target_class,'\',images(j).name);
                imwrite(image, char(image_save_path));          
            end

        end 
        
    end
    
end

%% Horizantally flip the images
function horizontalFlip(dataset_dir, target_dir, prob)
    subsets = dir(dataset_dir);
    numOfSubsets = size(subsets,1);

    for k=3:numOfSubsets
        subset_path = strcat(dataset_dir,'\',subsets(k).name);
        classes = dir(subset_path);
        numOfClasses = size(classes,1);

        target_subset = strcat(target_dir,'\', subsets(k).name);
        if isdir(char(target_subset)) ~= 1
            mkdir(char(target_subset))
        end
        for i=3:numOfClasses

            class_path = strcat(subset_path,'\', classes(i).name);
            images = dir(class_path);

            target_class = strcat(target_subset,'\', classes(i).name);
            if isdir(char(target_class)) ~= 1
                mkdir(char(target_class))
            end
            for j=3:size(images,1);
                image_path = strcat(class_path,'\', images(j).name);
                image = imread(image_path);
                
                if rand() < prob
                    flipped = flip(image ,2);
                
                    image_save_path = strcat(target_class,'\flipped_',images(j).name);
                    imwrite(flipped, char(image_save_path)); 
                end
                
            end
        end  
    end
end


%% Rotate the images
function rotateImages(dataset_dir, target_dir, angles)
    subsets = dir(dataset_dir);
    numOfSubsets = size(subsets,1);

    for k=3:numOfSubsets
        subset_path = strcat(dataset_dir,'\',subsets(k).name);
        classes = dir(subset_path);
        numOfClasses = size(classes,1);

        target_subset = strcat(target_dir,'\', subsets(k).name);
        if isdir(char(target_subset)) ~= 1
            mkdir(char(target_subset))
        end
        for i=3:numOfClasses

            class_path = strcat(subset_path,'\', classes(i).name);
            images = dir(class_path);

            target_class = strcat(target_subset,'\', classes(i).name);
            if isdir(char(target_class)) ~= 1
                mkdir(char(target_class))
            end
            for j=3:size(images,1);
                image_path = strcat(class_path,'\', images(j).name);
                image = imread(image_path);
                
                angle = randsample(angles, 1);
                rotated = imrotate(image, angle, 'bilinear', 'crop');
                
                image_save_path = strcat(target_class,'\rotated_',images(j).name);
                imwrite(rotated, char(image_save_path)); 
 
            end
            
        end  
    end
end


%% Rotate the images
function translate(dataset_dir, target_dir, amounts)
    subsets = dir(dataset_dir);
    numOfSubsets = size(subsets,1);

    for k=3:numOfSubsets
        subset_path = strcat(dataset_dir,'\',subsets(k).name);
        classes = dir(subset_path);
        numOfClasses = size(classes,1);

        target_subset = strcat(target_dir,'\', subsets(k).name);
        if isdir(char(target_subset)) ~= 1
            mkdir(char(target_subset))
        end
        for i=3:numOfClasses

            class_path = strcat(subset_path,'\', classes(i).name);
            images = dir(class_path);

            target_class = strcat(target_subset,'\', classes(i).name);
            if isdir(char(target_class)) ~= 1
                mkdir(char(target_class))
            end
            for j=3:size(images,1);
                image_path = strcat(class_path,'\', images(j).name);
                image = imread(image_path);
                
                amount = randsample(amounts, 1);
                if rand() < 0.5 % translate horizantally
                    if amount > 0
                        translated = image(:,[end:-1:end-amount+1,1:end-amount],:);
                    else
                        translated = image(:,[-amount+1:end,-amount:-1:1],:);
                    end
                else % translate vertically
                     if amount > 0
                        translated = image([end:-1:end-amount+1,1:end-amount],:,:);
                    else
                        translated = image([-amount+1:end,-amount:-1:1],:, :);
                    end                   
                end
                
                image_save_path = strcat(target_class,'\translated_',images(j).name);
                imwrite(translated, char(image_save_path)); 
 
            end
            
        end  
    end
end


%% Zoom the images
function zoomIn(dataset_dir, target_dir, factors)
    subsets = dir(dataset_dir);
    numOfSubsets = size(subsets,1);

    for k=3:numOfSubsets
        subset_path = strcat(dataset_dir,'\',subsets(k).name);
        classes = dir(subset_path);
        numOfClasses = size(classes,1);

        target_subset = strcat(target_dir,'\', subsets(k).name);
        if isdir(char(target_subset)) ~= 1
            mkdir(char(target_subset))
        end
        for i=3:numOfClasses

            class_path = strcat(subset_path,'\', classes(i).name);
            images = dir(class_path);

            target_class = strcat(target_subset,'\', classes(i).name);
            if isdir(char(target_class)) ~= 1
                mkdir(char(target_class))
            end
            for j=3:size(images,1);
                image_path = strcat(class_path,'\', images(j).name);
                image = imread(image_path);
                
                [h, w, d] = size(image); 
                factor = randsample(factors, 1);
                zoomed = imresize(image,factor);
                [h2, w2, d2] = size(zoomed);
                rec = [floor((h2-h)/2), floor((w2-w)/2), h-1,  w-1];
                zoomed = imcrop(zoomed, rec);
                
                image_save_path = strcat(target_class,'\zoomed_',images(j).name);
                imwrite(zoomed, char(image_save_path)); 
 
            end
            
        end  
    end
end


%% Resize the all images
function resizeAll(dataset_dir, target_dir, newSize)
    subsets = dir(dataset_dir);
    numOfSubsets = size(subsets,1);

    counter = 0;
    for k=3:numOfSubsets
        subset_path = strcat(dataset_dir,'\',subsets(k).name);
        classes = dir(subset_path);
        numOfClasses = size(classes,1);

        target_subset = strcat(target_dir,'\', subsets(k).name);
        if isdir(char(target_subset)) ~= 1
            mkdir(char(target_subset))
        end
        for i=3:numOfClasses

            class_path = strcat(subset_path,'\', classes(i).name);
            images = dir(class_path);

            target_class = strcat(target_subset,'\', classes(i).name);
            if isdir(char(target_class)) ~= 1
                mkdir(char(target_class))
            end
            for j=3:size(images,1);
                image_path = strcat(class_path,'\', images(j).name);
                image = imread(image_path);
                counter = counter + 1;
                
                resized = imresize(image, newSize);
                
                image_save_path = strcat(target_class,'\',images(j).name);
                imwrite(resized, char(image_save_path)); 
 
            end
            
        end  
    end
    fprintf('Total number of images  : %d\n', counter);
end












