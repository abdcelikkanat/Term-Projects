function utility(  )

% generateDataset();

generateImageIndices()

generateClassLabels()


end

function generateClassLabels()
    source_path = 'C:\Users\Abdulkadir\Desktop\Datasets\IITD';
    
    load('mat_files/discardedImageIndices.mat');   
    discardedImages = discardedImageIndices;

    images = dir(source_path);
    N = size(images,1); % number of images + 2 for '.' and '..'    
    index = 1:N;
    
    userIds = zeros(N,1);
    for i=3:N;
        imageName = images(i).name;
        userIds(i) = str2double(imageName(1:3));
    end
    classLabels = userIds(~ismember(index, discardedImages));
    classLabels = classLabels(3:end);
    save('./mat_files/IITD_classLabels.mat','classLabels')

end

function generateImageIndices()
    source_path = 'C:\Users\Abdulkadir\Desktop\Datasets\IITD';
    
    discardedImageIndices = [147, 164, 254, 623, 781, 988, 990, 991, 992,...
        993, 1012, 1013, 1014, 1015, 1016, 1017, 1029, 1099, 1103, 1104,...
        1105, 1215, 1216];
    
    numOfSubjects = 230;

    images = dir(source_path);
    N = size(images,1); % number of images + 2 for '.' and '..'

    

    imageIds = zeros(N,1);
    userIds = zeros(N,1);
    for i=3:N;

        imageName = images(i).name;
        imageIds(i) = str2double(imageName(end-4:end-4));
        userIds(i) = str2double(imageName(1:3));
        
    end    
    v = userIds;
    v(discardedImageIndices) = 0;

    % FÝnd the subjects that containing less than ? images and discard
    % their hand images
    discardedSubjectIndices = [];
    for person=1:numOfSubjects
        if sum(v == person) < 3
            discardedSubjectIndices = [discardedSubjectIndices, person];
        end
    end

    index = 1:N;
    inx = index(~ismember(index, discardedImageIndices));
    inx = inx(2:end);
    for i=inx

        if ismember(userIds(i), discardedSubjectIndices) == 1
            discardedImageIndices = [discardedImageIndices, i];
        end
    end


    save('./mat_files/discardedImageIndices.mat','discardedImageIndices')

end