function process_log_files( )
    blue = [0, 0.4470, 0.7410];
    red = [0.8500, 0.3250, 0.0980];
    yellow = [0.9290, 0.6940, 0.1250];
    purple = [0.4940, 0.1840, 0.5560];
    green = [0.4660, 0.6740, 0.1880];
    
    base = '../log/';
    
    filename = 'log_pet_10class.txt';
    [training1, test1] = read_log_file(base, filename);
    filename = 'log_pet_10class_ideal2.txt';
    [training2, test2] = read_log_file(base, filename);
    filename = 'log_smaller_kernels_pet_10class.txt';
    [training3, test3] = read_log_file(base, filename);
    
    figure, hold on
    len = 200;
    training1 = training1(1:len); test1 = test1(1:len);
    h1 = plot(1:numel(training1),training1,'-','Color', blue);
    h2 = plot(1:numel(test1),test1,'--','Color', blue);
    
    training2 = training2(1:len); test2 = test2(1:len);
    h3 = plot(1:numel(training2),training2,'-','Color', red);
    h4 = plot(1:numel(test2),test2,'--','Color', red);
    
    training3 = training3(1:len); test3 = test3(1:len);
    h5 = plot(1:numel(training3),training3,'-','Color', green);
    h6 = plot(1:numel(test3),test3,'--','Color', green);
    title('The Effect of The Number Of Classes')
    legend([h1, h3, h5, h2, h4, h6], 'M-CNN (Train)','Dropout (Train)','Additional (Train)',...
        'M-CNN (Test)','Dropout (Test)','Additional (Test)')
    xlabel('Epochs'), ylabel('Error %')
    %legend('Training (2 class)','Test (2 class)',...
    %        'Training (10 class)','Test (10 class)',...
    %        'Training (37 class)','Test (37 class)')
    fprintf('1: Min training : %f test : %f \n',min(training1(1:len)), min(test1(1:len)))
    fprintf('2: Min training : %f test : %f \n',min(training2(1:len)), min(test2(1:len)))
    fprintf('3: Min training : %f test : %f \n',min(training3(1:len)), min(test3(1:len)))
end


function process_log_files_2( )
    blue = [0, 0.4470, 0.7410];
    red = [0.8500, 0.3250, 0.0980];
    yellow = [0.9290, 0.6940, 0.1250];
    purple = [0.4940, 0.1840, 0.5560];
    green = [0.4660, 0.6740, 0.1880];
    
    base = '../log/';
    
    filename = 'log_pet_10class.txt';
    [training1, test1] = read_log_file(base, filename);
    filename = 'log_pet_10class_more.txt';
    [training2, test2] = read_log_file(base, filename);
    filename = 'log_pet_10class_less.txt';
    [training3, test3] = read_log_file(base, filename);
    
    figure, hold on
    len = 200;
    training1 = training1(1:len); test1 = test1(1:len);
    h1 = plot(1:numel(training1),training1,'-','Color', blue);
    h2 = plot(1:numel(test1),test1,'--','Color', blue);
    
    training2 = training2(1:len); test2 = test2(1:len);
    h3 = plot(1:numel(training2),training2,'-','Color', red);
    h4 = plot(1:numel(test2),test2,'--','Color', red);
    
    training3 = training3(1:len); test3 = test3(1:len);
    h5 = plot(1:numel(training3),training3,'-','Color', green);
    h6 = plot(1:numel(test3),test3,'--','Color', green);
    %title('The Effect of The Number Of Classes')
    legend([h1, h3, h5, h2, h4, h6], 'M-CNN (Train)','Smaller (Train)','Larger (Train)',...
        'M-CNN (Test)','Smaller (Test)','Larger (Test)')
    xlabel('Epochs'), ylabel('Error %')
    %legend('Training (2 class)','Test (2 class)',...
    %        'Training (10 class)','Test (10 class)',...
    %        'Training (37 class)','Test (37 class)')
    fprintf('1: Min training : %f test : %f \n',min(training1(1:len)), min(test1(1:len)))
    fprintf('2: Min training : %f test : %f \n',min(training2(1:len)), min(test2(1:len)))
    fprintf('3: Min training : %f test : %f \n',min(training3(1:len)), min(test3(1:len)))
end

function [training, test] = read_log_file(base, filename)
    file_dir = strcat(base,filename);

    fileID = fopen(file_dir,'r');
    
    tline = fgetl(fileID);
    counter = 1;
    training = []; test = [];
    while ischar(tline)
        if counter > 3 && mod(counter-2,3) == 2
            %disp(tline)
            tokens = strsplit(tline,' ');
            training = [training; str2double(tokens(3))];
        end
        if counter > 3 && mod(counter-2,3) == 0
            %disp(tline)
            tokens = strsplit(tline,' ');
            test = [test; str2double(tokens(3))];
        end
        tline = fgetl(fileID);
        counter = counter + 1;
    end
    fclose(fileID);    

end
