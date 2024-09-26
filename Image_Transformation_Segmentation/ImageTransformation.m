clear; clc; close all;
%Read atlas - will b the floating image
atlas_img = imread("atlas.tiff");
%Read brain image - will be the reference image
brain_img = imread("brain.tiff");
%Q1
affine(brain_img,atlas_img,false,10,"Q1")
%Q2
affine(brain_img,atlas_img,false,10,"Q2")
%Q6
affine(brain_img,atlas_img,true,8,"Q6")

%% Thiss function performed the affine function
function affine(ref_img,floating_img,reject,threshold,name)
    %parameter estimation part
    %Create homogenous matrix equation D
    %Select Points dynamically by using the cpselect function
   [selectedMovingPoints,selectedFixedPoints]  = cpselect(floating_img,ref_img,'Wait',true)
   [num_selected_point, temp] = size(selectedFixedPoints)
   if num_selected_point >= 4
   save("selectedMovingPoints"+name+".mat","selectedMovingPoints")
   save("selectedFixedPoints"+name+".mat",'selectedFixedPoints')
   %create control points positions array as in the slide by using the
   %selected functions
   xw = [];
   yw =[];
   xu = [];
   yu = [];
   for h=1:num_selected_point
       xw =[xw;selectedMovingPoints(h,1) ];
       yw = [yw; selectedMovingPoints(h,2) ];
       xu = [xu; selectedFixedPoints(h,1)];
       yu = [yu; selectedFixedPoints(h,2)];
   end
  
    %% Purposely enter a control point pair which does not well match.
    % Suggest an algorithm for identifying a bad control point pair and rejecting it automatically to
    %achieve good registration
    distances_selected = sqrt((xw - xu).^2 + (yw - yu).^2)
    average_distance_selected = mean(distances_selected)
    if reject & num_selected_point >= 5
        std_of_distances = std(distances_selected)
        diff_of_distance = abs(distances_selected - average_distance_selected)
        if threshold < std_of_distances %then reject the outlier point
            [max_dist, index] = max(diff_of_distance)
            labels = "("+xw+","+yw+")-("+xu+","+yu+")"
            x = 1:length(distances_selected)
            figure
            % Plot
            plot(x, distances_selected, 'o');
            
            % Add labels for x-axis
            set(gca, 'XTick', x, 'XTickLabel', labels);
            xlabel('Categories');
            
            % Add labels for y-axis and title
            ylabel('Values');
            title('Plot with String and Numerical Values');
            % Draw the mean line
            hold on; % Hold the current plot
            mean_line = refline(0, average_distance_selected); % Create a horizontal line at the mean distance
            mean_line.Color = 'r'; % Set the color of the line to red
            legend(mean_line, 'Mean Distance'); % Add a legend for the mean line
            hold off; % Release the current plot
            %remove the points from the selected array
            xw(index) = []
            yw(index) = []
            xu(index) = []
            yu(index) = []
        end
    end    
    %Define D which is entirely specified from contorl points
    D = [];
    for i= 1:length(xu)
        temp_mat = [1 xu(i) yu(i) xu(i).*yu(i)];
        D = [D; temp_mat];
    end
    if length(xu) <= 4 % Simple Affine part with 4 control points
        inv_D = inv(D);
        A = inv_D* xw;
        B = inv_D* yw;
    else % Overestimated Affine part with more than 4 control points, use pseudo-inverse - Q2
        pinv_D = pinv(D);
        A = pinv_D* xw;
        B = pinv_D* yw;
    end
    %Show the estimated parameters
    disp('Estimated Parameters A:');
    disp(A);
    disp('Estimated Parameters B:');
    disp(B);
    % Measure the quality of the match - Q3
    distances = sqrt((xw - D*A).^2 + (yw - D*B).^2)
    average_distance = mean(distances)
    fprintf('Average Euclidean distance between corresponding control points: %.2f pixels\n', average_distance);
    % Backward Transformation Part
    [rows, cols] = size(floating_img);
    output_img = zeros(size(floating_img));
    for i = 1:rows
        for j = 1:cols
            % Compute corresponding coordinates in reference image
            x_ref = A(1) + A(2)*j + A(3)*i + A(4)*i*j;
            y_ref = B(1) + B(2)*j + B(3)*i + B(4)*i*j;

            % Perform bilinear interpolation
            x_int = floor(x_ref);
            y_int = floor(y_ref);
            dx = x_ref - x_int;
            dy = y_ref - y_int;

            % Check boundaries
            if x_int >= 1 && x_int < cols && y_int >= 1 && y_int < rows
                %Perform bilinear interpolation
                output_img(i,j) = (1-dy) * (1-dx) * floating_img(y_int,x_int) + ...
                    (1-dy) * dx * floating_img(y_int+1,x_int)+...
                    dy*(1-dx)* floating_img(y_int,x_int+1)+...
                    dy*dx* floating_img(y_int+1,x_int+1);

            end
        end
    end
    %Fuse the images
    final_img = imfuse(uint8(output_img),ref_img,'blend');
    %Q5 color overlayed version
    colored_img = imfuse(output_img,ref_img, 'ColorChannels',[0 2 1]);
    images = {ref_img,floating_img, output_img, final_img,colored_img};
    labels = {'Brain Image','Atlas Image',"Atlas Image After Affine with " + length(xw) + " control points"," Fused Image", "Colored Overlay"};
    showImages(images,labels,'Resulting_Images',"Atlas Image After Affine with " + length(xw) + " control points "+ name, 3,2);
   else
       disp('Please select at least 4 points')
   end
end
%% This function will display the images 
function showImages(images, labels, file_name, final_image_name, subplot_x,subplot_y)
%Display the Images 
    figure
    for i=1:length(images)
        currentImage = images{i};
        % Create a subplot
        subplot(subplot_x, subplot_y, i);
        % Display the image with its label
        imshow(currentImage, []);
        % Compute the noise standard deviation and mean before and after filtering. 
        title(labels{i});
        imwrite(mat2gray(currentImage),fullfile(file_name, labels{i}+".jpg"));
    end
    % Adjust layout
    sgtitle(final_image_name);
    set(gcf, 'Position', [100, 100, 800, 600]);
    saveas(gcf, fullfile(file_name, final_image_name+ ".jpg"));
end
