clear; clc; close all;
% Read Contrast1_new.tif
contrast1_new = imread("Contrast1_new.tif");
% Read Contrast2_new.tif -- is a translated version of Contrast1-new by 1.8
% pixels and 2.1 pixels in the x and y directions, respectively
contrast2_new = imread("Contrast2_new.tif");

% d- The first frame should be the difference
% image before any translation and the last frame should be the difference image
% after registration has been optimized.
v = VideoWriter('Q3_Resulting_Videos/Q3optimize_myNCC_1_1.avi')
v1 = VideoWriter('Q3_Resulting_Videos/Q3optimize_myNCC_0_0.avi')
v2 = VideoWriter('Q3_Resulting_Videos/Q3optimize_SSE_1_1.avi')
v3 = VideoWriter('Q3_Resulting_Videos/Q3optimize_SSE_0_0.avi')
% Make  video frame rate 1 so you can easily see the changes
v.FrameRate = 1;
v1.FrameRate = 1;
v2.FrameRate = 1;
v3.FrameRate = 1;
%Open the videos
open(v)
open(v1)
open(v2)
open(v3)
% b- iteratively minimize the cost function. Start from a reasonable initialization, say (0,0)
%start at dx=0, dy=0 if only translation is used
p0 = [0 0];
% define options e.g. (doc optimset for details)
options = optimset('Display','iter');
%subtracted image without registration
image_subtracted_without_reg = contrast1_new - contrast2_new;
writeToVideo(image_subtracted_without_reg,v,"Initial subtraction without the registration");
writeToVideo(image_subtracted_without_reg,v1,"Initial subtraction without the registration");
writeToVideo(image_subtracted_without_reg,v2,"Initial subtraction without the registration");
writeToVideo(image_subtracted_without_reg,v3,"Initial subtraction without the registration");
% actual call to optimization
% I used both SSE and myNCC as the similarity cost function
% I used both (0,0) and (1,1) as the initializing point to observe the
% difference
[phat] = fminsearch(@funregister,p0,options,contrast1_new,contrast2_new,v1,"myNCC");
[phat2] = fminsearch(@funregister,p0,options,contrast1_new,contrast2_new,v3,"SSE");
p1 = [1 1];
[phat3] = fminsearch(@funregister,p1,options,contrast1_new,contrast2_new,v,"myNCC");
[phat4] = fminsearch(@funregister,p1,options,contrast1_new,contrast2_new,v2,"SSE");
% c- Display Contrast1_new side by-byside
% with translated Contrast2_new and report the "optimal" translation in x and y
% yielded by the algorithm.
[error_MyNCC_p0, image2_registered_MyNCC_p0] = funregisterdisplay(phat,contrast1_new,contrast2_new, "myNCC");
[error_SSE_p0, image2_registered_SSE_p0] = funregisterdisplay(phat2,contrast1_new,contrast2_new, "SSE");


[error_MyNCC_p1, image2_registered_MyNCC_p1] = funregisterdisplay(phat3,contrast1_new,contrast2_new, "myNCC");
[error_SSE_p1, image2_registered_SSE_p1] = funregisterdisplay(phat4,contrast1_new,contrast2_new, "SSE");

% d- The first frame should be the difference
% image before any translation and the last frame should be the difference image
% after registration has been optimized.
%find subtracted images
image2_subtracted_SSE_p0 = contrast1_new - image2_registered_SSE_p0;
writeToVideo(image2_subtracted_SSE_p0 , v3,"Final Optimal One p: ("+ phat2(1)+","+phat2(2)+")" )
close(v3);
image2_subtracted_MyNCC_p0 = contrast1_new - image2_registered_MyNCC_p0;
writeToVideo(image2_subtracted_MyNCC_p0, v1,"Final Optimal One p: ("+ phat(1)+","+phat(2)+")" )
close(v1);
image2_subtracted_SSE_p1 = contrast1_new - image2_registered_SSE_p1;
writeToVideo(image2_subtracted_SSE_p1, v2,"Final Optimal One p: ("+ phat4(1)+","+phat4(2)+")"  )
close(v2);
image2_subtracted_MyNCC_p1 = contrast1_new - image2_registered_MyNCC_p1;
writeToVideo(image2_subtracted_MyNCC_p1, v,"Final Optimal One p: ("+ phat3(1)+","+phat3(2)+")" )
close(v);


images = {contrast1_new, image2_registered_MyNCC_p0, image_subtracted_without_reg,image2_subtracted_MyNCC_p0, ...
   contrast1_new,  image2_registered_SSE_p0,image_subtracted_without_reg, image2_subtracted_SSE_p0, ...
   contrast1_new, image2_registered_MyNCC_p1,image_subtracted_without_reg,  image2_subtracted_MyNCC_p1, ...
   contrast1_new,  image2_registered_SSE_p1,image_subtracted_without_reg,image2_subtracted_SSE_p1};

labels = {"Contrast 1", "Contrast 2 registered with 0,0 and myNCC, error: " + error_MyNCC_p0,"Subtracted image without registration", "Subtracted image myNCC with p0 = 0,0 optimal p="+ phat(1)+","+phat(2), ... 
    "Contrast 1", "Contrast 2 registered with 0,0 and SSE, error: " + error_SSE_p0,"Subtracted image without registration",  "Subtracted image SEE with p0 = 0,0 optimal p="+ phat2(1)+","+phat2(2), ...
    "Contrast 1", "Contrast 2 registered with 1,1 and myNCC, error: " + error_MyNCC_p1,"Subtracted image without registration",  "Subtracted image myNCC with p1= 1,1 optimal p="+ phat3(1)+","+phat3(2), ...
    "Contrast 1", "Contrast 2 registered with 1,1 and SSE, error: " + error_SSE_p0, "Subtracted image without registration", "Subtracted image SSE with p1=1,1 optimal p="+ phat4(1)+","+phat4(2)};

showImages(images,labels,'Q3_Resulting_Images',"C- Display of registration", 4,4);

%% This function will be used to optimize the registration so that the similarity cost function become the highest
function [error] = funregister(p,image1,image2, v, similarity_cost_function_name)
    disp("p (" + p(1) + "," +p(2) + ")")
    x = p(1);
    y = p(2);
    %defines transformations - translation matrix (I applied 2d affine func)
    trans = affine2d([1 0 0; 0 1 0; x y 1]);
    
    %initializes imref2d object
    outView = imref2d([size(image1),size(image1)]);
    
    % To see the translation, both the filled values black and white is defined
    % with imwarp, cubic interpolation is used
    image2_registered = imwarp(image2,trans,'cubic','OutputView',outView,'FillValues',255);
    
    % a- Used your function, myNCC
    if similarity_cost_function_name == "myNCC"
    % Compute myNCC after registration
    similarity_registered = myNCC(image1,image2_registered)
    error = 1-similarity_registered
    subtracted_img = image1 - image2_registered;
    writeToVideo(subtracted_img,v,"("+p(1)+","+p(2)+") error: "+ error); %for part d
    % b- Use x and y translation only (no rotation, no scaling, no shearing). You will need to
    % evaluate a similarity cost function (that is, SSE)
    elseif similarity_cost_function_name == "SSE"
        error = sumsqr(image1-image2_registered)
        subtracted_img = image1 - image2_registered;
        writeToVideo(subtracted_img,v,"("+p(1)+","+p(2)+") error: "+error); %for part d
    else
        fprintf('You need to give the name of the similarity cost function');
    end
end


%I copied the optimization function but, I did not use it to optimize. I
%used this function to get the error and the image after the registration
%value of the specific function
function [error, image2_registered] = funregisterdisplay(p,image1,image2, similarity_cost_function_name)
    disp("p (" + p(1) + "," +p(2) + ")")
    x = p(1);
    y = p(2);
    %defines transformations - translation matrix (I applied 2d affine func)
    trans = affine2d([1 0 0; 0 1 0; x y 1]);
    
    %initializes imref2d object
    outView = imref2d([size(image1),size(image1)]);
    
    % To see the translation, both the filled values black and white is defined
    % with imwarp, cubic interpolation is used
    image2_registered = imwarp(image2,trans,'cubic','OutputView',outView,'FillValues',255);
    % a- Used your function, myNCC
    if similarity_cost_function_name == "myNCC"
    % Compute myNCC after registration
    similarity_registered = myNCC(image1,image2_registered)
    error = 1-similarity_registered
    % b- Use x and y translation only (no rotation, no scaling, no shearing). You will need to
    % evaluate a similarity cost function (that is, SSE)
    elseif similarity_cost_function_name == "SSE"
        error = sumsqr(image1-image2_registered)
    else
        fprintf('You need to give the name of the similarity cost function');
    end
end

% This function will display the images 
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

% d- At each step in the optimization algorithm (when each new Tx and Ty is computed),
% please using MATLAB%s getframe() to iteratively capture the difference image between
% the fixed image and the translated image.
function writeToVideo(image, video, label)
    % Create a figure without displaying it
    fig = figure('Visible', 'off');
    subplot(1, 1, 1);
    % Display the image with its label
    imshow(image, []);
    % Compute the noise standard deviation and mean before and after filtering. 
    title(label);
    writeVideo(video, getframe(fig));
    % for i = 1:10 % Repeat the first frame 10 times to slow down the frame rate
    %     writeVideo(video, getframe(fig));
    % end
    close(fig);
end




