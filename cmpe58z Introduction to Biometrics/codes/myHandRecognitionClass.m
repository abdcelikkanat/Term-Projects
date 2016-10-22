classdef myHandRecognitionClass
    properties
      Value
    end
    methods
    function [output] = applyBinarization(~, image, threshold_level, v)        
        image = rgb2gray(image);
        image = imadjust(image);
        if threshold_level == -1 % -1 means, calculate its level
            level = graythresh(image);
        else
            level = threshold_level;
        end
        image = im2bw(image, level);
        output = image;
        if v == 1
            figure, imshow(image), hold on, title('Binarization')
        end
        
    end 
    
    % Extract hand from background
    function [output] = applyMorphologicalOps(~, image, type, v)
        se3 = strel('disk',3);
        se5 = strel('disk',5);
        se7 = strel('disk',7);
        se9 = strel('disk',9);
        se11 = strel('disk',11); 
        
        % Determine the which type of morph. ops. will be used
        % 1 is used for IITD and 2 is used for bosphorus
        if type == 1
            
            img = medfilt2(image,[7 7]);
            img = imerode(img,se7);
            img = imdilate(img,se5);

            img = imdilate(img,se9);
            img = imerode(img,se9);

            img = imdilate(img,se5);
            img = imerode(img,se5);

            img = imerode(img,se3);
            img = imdilate(img,se3);
 
            img = medfilt2(img);

            img = medfilt2(img, [15, 15]);
     
            img = imdilate(img,se11);
            img = imerode(img,se11);

            img = imdilate(img,se11);

            img = medfilt2(img, [17, 17]);
  
            img = imerode(img,se11);

            img = imerode(img,se5);
            img = imdilate(img,se5);            
              
        elseif type == 2

            img = medfilt2(image,[5, 5]);
            img = imerode(img,se3);
            img = imdilate(img,se3);
            
        end
        
        output = img;
        if v == 1
            figure, imshow(output), hold on, title('Morph. Ops')
        end
        
    end
    
    % Extracts the hand, and if a finger is disconnected, connects it to hand    
    function [output] = extractHand(~, input, v)

        image = bwpropfilt(input,'perimeter',1);
        output = image;
        
        image2 = bwpropfilt(input,'FilledArea',2);
        area = sum(sum(image2-image)); % size of the second connected component
        
        if area > numel(image)*0.005
            % if second connected component larger than certain theresholda
            s = regionprops(image2-image,'centroid');
            centroid = cat(1, s.Centroid);
            l = find(sum(image,1)>0, 1, 'first');
            r = find(sum(image,1)>0, 1, 'last');
            u = find(sum(image,2)>0, 1, 'first');
            d = find(sum(image,2)>0, 1, 'last');
            
            if v == 1
                figure, imshow(image2), hold on, title('Hand extraction 1')
                plot(centroid(1),centroid(2),'ro', r, u, 'b*');
            end
            
            % Connect the disconnected finger to the hand if exists
            if d > centroid(2) & u < centroid(2) & l < centroid(1) & r > centroid(1) 

                finger = image2-image;
                % Find the orientation of the finger
                stats = regionprops(finger,'orientation','MajorAxisLength');
                orient = stats.Orientation;
                if (90 >= orient) && (orient >= 0)
                    angle = 90-orient;
                end
                if (0 > orient) && (orient >= -90)
                    angle = -90-orient;
                end

                % Rotate the finger 
                rotatedFinger = imrotate(finger,angle,'nearest','crop');

                % Extend the disconnected finger
                se = strel('line',40,91);
                dilatedFinger = imdilate(rotatedFinger,se);
                % Rotate the finger back
                extendedFinger = imrotate(dilatedFinger, -angle,'nearest','crop');
                % Connect the finger and hand
                output = image + extendedFinger;
                output(output==2) = 1;
                
                if v ==  1
                    figure, imshow(output), hold on, title('Hand extraction 2')
                end
                
            end
            
        end
 
    end
    
    % Rotate the hand image with respect to major axis
    function [rotated] = rotateHandImage(~, image, v)

        stats = regionprops(image,'orientation','MajorAxisLength');
        orient = stats.Orientation;
        if (90 >= orient) && (orient >= 0)
            angle = 90-orient;
        end
        if (0 > orient) && (orient >= -90)
            angle = -90-orient;
        end

        % Rotate image
        rotated = imrotate(image,angle,'nearest','crop');

        if v == 1
            figure, imshow(rotated), hold on, title('Rotated image')
        end
        
    end
    
    
    function [r] = findReferencePoint(~, binaryImage, v)
        % Find centroid
        s = regionprops(binaryImage,'centroid');
        centroid = cat(1, s.Centroid);

        % Find reference point
        r = zeros(2,1);
        j = 0;
        [r, c] = size(binaryImage);
        tmp = zeros(1,c);

        for i=1:r
            if binaryImage(i,floor(centroid(1))) == 1
                j = j + 1;
                tmp(j) = i;
            end
            r(1) = max(tmp);
            r(2) = centroid(1);
        end
    
        if v == 1
            figure, imshow(binaryImage), hold on,, title('Reference point')
            plot(centroid(1),centroid(2),'b*','LineWidth',2)
            plot(r(2), r(1), 'r*','LineWidth',2)
        end

    end
    
    % Extract the hand contour
    function [boundary] = extractHandContour(~, image, v)
    
        [B,L] = bwboundaries(image,'noholes');
        if v == 1
            figure, imshow(L), title('Extracted Hand Contour'), hold on
        end
        for k = 1:length(B)
           boundary = B{k};
           if v == 1
               plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
           end
        end

    end
    
    
    function [peaks, peakInx, valleys, valleyInx, composite, compositeInx, midPoints] = extractFingerPoints(~, image, contour, r, v) 
    
        distH = contour(:,1) - r(1);
        distW = contour(:,2) - r(2);
        dist = sqrt(distH.^2 + distW.^2);

        % Smooth the distance map
        y = medfilt1(dist,50);

        if v == 1
            figure, plot(1:numel(dist),y,'b.')
        end

        % Find local maxima and minima
        a = 1:numel(dist);
        [pks,locs_peak] = findpeaks(y);

        % if the peak points are on a line segment, take the center of line segment
        for n=1:numel(locs_peak)
            o = 0;
           while(y(locs_peak(n)) == y(locs_peak(n)+o))
                o = o+1;
           end
           locs_peak(n) = locs_peak(n)+floor(o/2);
        end

        if v == 1
            hold on,  plot(a(locs_peak),y(locs_peak),'r*','LineWidth',2)
        end

        peaks(1,:) = contour(locs_peak(1:5),2);
        peaks(2,:) = contour(locs_peak(1:5),1);
        peakInx = locs_peak(1:5);

        [vly, locs_valley] = findpeaks(-y(locs_peak(1):locs_peak(5)));

        % if the valley points are on a line segment, take the center of line segment
        for n=1:numel(locs_valley)
            o = 0;
           while(y(locs_peak(1)-1+locs_valley(n)) == y(locs_peak(1)-1+locs_valley(n)+o))
                o = o+1;
           end
           locs_valley(n) = locs_valley(n)+floor(o/2);
        end

        % Find the difference between locs(1) and starting point of boundary
        diff = a((locs_peak(1)-1)); 

        if v == 1
            plot(diff+locs_valley,y(diff+locs_valley),'y*','LineWidth',2);
        end

        valleys(1,:) = contour(diff+locs_valley(1:4),2);
        valleys(2,:) = contour(diff+locs_valley(1:4),1);
        valleyInx = diff+locs_valley(1:4);

        %% Extract composite points
        com_inx = zeros(1,5);
        % composite points for 2,3, and 4
        com_inx(2:4) = a(locs_peak(2:4)) - (a(locs_valley(2:4)) - a(locs_peak(2:4)));
        % composite point for 1
        com_inx(1) = numel(a) + a(locs_peak(1)) - (a(locs_valley(1)) - a(locs_peak(1)));
        % composite point for 5
        com_inx(5) = a(locs_peak(5)) + (a(locs_peak(5)) - a(locs_valley(4)));
        composite = [contour(com_inx,2)'; contour(com_inx,1)'];

        %% Extract mid-points
        midPoints = (composite + [valleys, valleys(:,4)]) / 2;

        compositeInx = com_inx;
       
        if v == 1
            figure, imshow(image), hold on
            plot(peaks(1,:), peaks(2,:),'rx', 'LineWidth',2)
            plot(valleys(1,:),valleys(2,:),'bx', 'LineWidth',2)
            plot(composite(1,:),composite(2,:),'gx', 'LineWidth',2)
            plot(midPoints(1,:),midPoints(2,:),'mx', 'LineWidth',2) 
        end

    end
    
    
    function [output, cutboundary] = cutWrist(~, image, contour, peaks, valleys, peakInx, compositeInx, v)

        t = floor(1.75*(compositeInx(5)-peakInx(5)))+ peakInx(5);
        if t > size(contour,1)
            t = t - size(contour,1);
        end

        u = peakInx(1) - floor(1.75*(size(contour,1)-compositeInx(1) + peakInx(1)));
        if(u <= 0)
           u = size(contour,1) + u;
        end
        
        if v == 1
            figure, imshow(image), hold on, title('Wrist Cut')
            plot(contour(u,2), contour(u,1), 'm*','LineWidth', 2)
            plot(contour(t,2), contour(t,1), 'm*','LineWidth', 2)
        end
        output = logical(image);

        output(contour(u,1):end,:) = 0;
        if v == 1
            figure, imshow(output), hold on
        end
        output = bwpropfilt(output,'perimeter',1);

        [B,L] = bwboundaries(output,'noholes');
        if v == 1
            figure, imshow(L), hold on
        end
        for k = 1:length(B)
           boundary = B{k};

        end

        cutboundary = [];
        for i=1:size(boundary,1)
            if boundary(i,1) < contour(u,1)-2
               cutboundary = [cutboundary; boundary(i,:)];
            else

            end
        end
        if v == 1
            plot(cutboundary(:,2), cutboundary(:,1), 'r.', 'LineWidth', 2)
        end
    end
    
    % This function also rotates hand image with respect to the middle
    % finger
    function [output] = rotateHandImage2(~, image, peak, midPoint, v)
        % peak and midPoint is the coordinates for middle finger
        if v == 1
            figure, imshow(image), hold on, title('Rotated 2')
            plot(peak(1,1),peak(2,1),'bo')
        end
        x = [midPoint(1,1), peak(1,1) ];
        y = [midPoint(2,1), peak(2,1)];
        if v == 1
            plot(x, y,'xr-')
        end
        angle = atand( (midPoint(2,1)-peak(2,1))/(midPoint(1,1)-peak(1,1)) );
        if angle < 0
            alpha = 90 + angle;
        else
            alpha = -90 + angle;
        end
        rotated = imrotate(image, alpha,'nearest','crop');
        if v == 2
            figure, imshow(rotated), hold on
            plot(midPoint(1,1),midPoint(2,1),'ro')
        end
        output = rotated;

    end
    
    % Register all fingers
    function [newContour] = registerFingers(~, image, contour, peaks, valleyInx, compositeInx, midPoints, verb)
        if verb == 1
            figure, imshow(image), hold on
            plot(contour(:,2),contour(:,1),'.', 'LineWidth', 2)
%             plot(midPoints(1,3), midPoints(2,3),'ro')
        end
        
        w = [60, 30, 10, -10, -20];

        phi = (90 - atand( (midPoints(2,:)-peaks(2,:))./(midPoints(1,:)-peaks(1,:)) ) );

        x = midPoints(1,2) + (contour(compositeInx(2):valleyInx(2), 2)-midPoints(1,2))*cosd(phi(2)) - ...
            (contour(compositeInx(2):valleyInx(2), 1)-midPoints(2,2))*sind(phi(2));

        y = midPoints(2,2) + (contour(compositeInx(2):valleyInx(2), 1)-midPoints(1,2))*sind(phi(2)) + ...
            (contour(compositeInx(2):valleyInx(2), 1)-midPoints(2,2))*cosd(phi(2));

        alpha = zeros(1,5);
        newContour = [];
        for i=1:5
           if phi(i) < 90
               alpha(i) = w(i) - phi(i);
           else
               alpha(i) = w(i) + (180-phi(i));
           end
           betwX = []; betwY = [];
           if i == 1
                u = contour([compositeInx(1):end, 1:valleyInx(i)], 2)'- midPoints(1,i);
                v = contour([compositeInx(i):end, 1:valleyInx(i)], 1)'- midPoints(2,i);
            
                betwX = contour(valleyInx(1):compositeInx(2), 2);
                betwY = contour(valleyInx(1):compositeInx(2), 1);

           elseif i == 5
                u = contour(valleyInx(4):compositeInx(5), 2)'- midPoints(1,i); 
                v = contour(valleyInx(4):compositeInx(5), 1)'- midPoints(2,i); 
    %            betwX = contour(compositeInx(5):compositeInx(1), 2);
    %            betwY = contour(compositeInx(5):compositeInx(1), 1);  
           else
                u = contour(compositeInx(i):valleyInx(i), 2)'- midPoints(1,i); 
                v = contour(compositeInx(i):valleyInx(i), 1)'- midPoints(2,i); 

    %            betwX = contour(valleyInx(i):compositeInx(i+1), 2);
    %            betwY = contour(valleyInx(i):compositeInx(i+1), 1);
           end

           R = [cosd(-alpha(i)), -sind(-alpha(i)); sind(-alpha(i)), cosd(-alpha(i))];
           z = R*[u; v];
           x = z(1,:) + midPoints(1,i);
           y = z(2,:) + midPoints(2,i);

           if i==1
               diffX = x(end) - (u(end) + midPoints(1,i));
               diffY = y(end) - (v(end) + midPoints(2,i));
           else
               diffX = x(1) - (u(1) + midPoints(1,i));
               diffY = y(1) - (v(1) + midPoints(2,i));
           end

            x = x - diffX;
            y = y - diffY;

            newContour = [newContour', [y; x]]';
            newContour = [newContour; [betwY, betwX]];
        end    
        
        if verb == 1
            display('AAAAAAAAAAA')
            figure, imshow(image), hold on
            plot(newContour(:,2),newContour(:,1),'m.','LineWidth', 2)
            
        end
    
    end
    
    % Shifts the contours to increase recognition performance
    function [shifted] = shiftContour(~, image, contour, midPoints, v)
        [h, w] = size(image); % The sizes are fixed for each image
        
        if v == 1
            figure, imshow(image), hold on
            plot(midPoints(1,:),midPoints(2,:),'rx')
        end
    
        center_x = mean(midPoints(1,2:5));
        center_y = mean(midPoints(2,2:5));
        
        if v == 1
            plot(center_x,center_y,'bo')
            plot(contour(:,2)+(w*0.6-center_x), contour(:,1)+(h*0.45-center_y),'m.')
                        
        end
        % Since the size of each image is fixed, take 0.6 and 0.45
        shifted = [contour(:,2)+(w*0.6-center_x), contour(:,1)+(h*0.45-center_y)];

    end
    
    %% Rescales the image such that minValue will  be the size of shortest length
    function [scaled] = scaleImage(~, image, minValue, v)
       
        [h, w, d] = size(image);
        if h > w
            scaled = imresize(image, [NaN minValue]);
        else
            scaled = imresize(image, [minValue, NaN]);
        end
        if v == 1
           figure, imshow(scaled), hold on, title('Scaled Image') 
        end
        
    end
    
    function [output] = enlargenImage(~, image, hor, ver, v)
        [h, w, d] = size(image);
        output = zeros(h+2*ver, w+2*hor);
        output(ver+1:ver+h, hor+1:hor+w) = image;
        output = logical(output);
        if v == 1
            figure, imshow(output), hold on, title('Enlarged Image')
        end
    end
    
    
   
    function [g6, g10, g21] = extractGeometricFeatures(~, peaks, midPoints)
    
        d = zeros(1,16);
        % Distances from 1 to 5
        d = sqrt( (peaks(1,:) - midPoints(1,:)).^2 + (peaks(2,:) - midPoints(2,:)).^2 );
        % Distance 6
        d(6) = sqrt( (midPoints(1,2)-midPoints(1,1))^2 + (midPoints(2,2)-midPoints(2,1))^2 );
        % Distance 7
        d(7) = sqrt( (midPoints(1,5)-midPoints(1,2))^2 + (midPoints(2,5)-midPoints(2,2))^2 ); 

        % Construct geometric feature vector
        g6 = [];
        for i=1:4
            for j=i+1:4
                g6 = [g6, d(i)/d(j)];
            end
        end
        
        g10 = [];
        for i=1:5
            for j=i+1:5
                g10 = [g10, d(i)/d(j)];
            end
        end
        
        g21 = [];
        for i=1:7
            for j=i+1:7
                g21 = [g21, d(i)/d(j)];
            end
        end

    end
    
    function [orientMap, distMap] = extractDistanceAndOrientationFeatures(~, contour, refPoint, size, v)
        ref_y = refPoint(1); ref_x = refPoint(2);

        if v == 1
            figure, hold on
            plot(contour(:,2), contour(:,1), 'r.')
            plot(ref_x, ref_y, 'bx')
        end
        distMap = sqrt( (contour(:,2) - ref_x).^2 + (contour(:,1) - ref_y).^2 );

        sigma = 1.e-10;
        orientMap = 90 + atand( ( contour(:,1)-ref_y )./( contour(:,2)-ref_x + sigma) );

        [c,l] = wavedec(orientMap,5,'db1');
        if size > l(end-1)
           error('Number of coefficients must be less than current given value'); 
        end
%         orientMap = c(end-l(end-1)+1:end-l(end-1)+50);
        
        
        [c,l] = wavedec(distMap,5,'db1');
 
%         distMap = c(end-l(end-1)+1:end-l(end-1)+50);

        distMap = c(l(1)+1:l(1)+50);
        orientMap = c(l(1)+1:l(1)+50);
        
        distMap = c(1:50);
        orientMap = c(1:50);
        
    end
    
    
   end
end
