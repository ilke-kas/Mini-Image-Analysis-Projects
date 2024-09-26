clear; clc; close all;
% Read Contrast1_new.tif
live_new = imread("live_new.tif");
% Read Contrast2_new.tif -- is a translated version of Contrast1-new by 1.8
% pixels and 2.1 pixels in the x and y directions, respectively
mask_new = imread("mask_new.tif");

% 3b- iteratively minimize the cost function. Start from a reasonable initialization, say (0,0)
%start at dx=0, dy=0 if only translation is used
p0 = [0.01 0.01];
p1 = [1 1];

% actual call to optimization
% 4-a Use different stopping criteria (the tolerances). Sample a range of values.
tolerances = [1e-4,1e-5,1e-6,1e-7];
table_1 = part4a(live_new, mask_new,tolerances,p0,"SSE");
table_2 = part4a(live_new, mask_new,tolerances,p1,"SSE");

table_3 =part4a(live_new, mask_new,tolerances,p0,"myNCC");
table_4 =part4a(live_new, mask_new,tolerances,p1,"myNCC");

table_part_a_cell = [table_1;table_2;table_3;table_4]
table_part_a = cell2table(table_part_a_cell, ...
    "VariableNames",["Similarity Method" "Initializing Point" "Tolerance" "Iteration Number" "Optimal Value" "Error"])

%% 4b- scalings
scalings = [1,5,10,20];
table_5 = part4b(live_new, mask_new,scalings,p0,"SSE")
table_6 = part4b(live_new, mask_new,scalings,p1,"SSE")

table_7 = part4b(live_new, mask_new,scalings,p0,"myNCC")
table_8 = part4b(live_new, mask_new,scalings,p1,"myNCC")

table_part_b_cell = [table_5;table_6;table_7;table_8]
table_part_b = cell2table(table_part_b_cell, ...
    "VariableNames",["Similarity Method" "Initializing Point" "Scaling" "Iteration Number" "Optimal Value" "Error"])
%% find the best ones from the tables 
[minValue, minIndex] = min(table_part_a.Error);
[minValue2, minIndex2] = min(table_part_b.Error);
%get the tolerance from table a
tolerance_optimal = table_part_a(minIndex,"Tolerance").Tolerance
scaling_optimal = table_part_b(minIndex2,"Scaling").Scaling
method = table_part_a(minIndex,"Similarity Method").("Similarity Method")
initializing_point = table_part_a(minIndex,"Initializing Point").("Initializing Point")
%% Perform again with these parameters
images = {};
lables = {};
video_name = "Q5DSA_Best"+method+"_"+initializing_point(1)+"_"+initializing_point(2)+"_"+scaling_optimal+"_"+tolerance_optimal+".avi"
v3 = VideoWriter('Q5_Resulting_Videos/'+video_name)
v3.FrameRate = 1;
open(v3)
%subtracted image without registration
image_subtracted_without_reg = live_new - mask_new;
 writeToVideo(image_subtracted_without_reg,v3,"Initial subtraction without the registration");
%Create options
% define options e.g. (doc optimset for details)
options = optimset('TolX',tolerance_optimal,'TolFun',tolerance_optimal,'PlotFcns', @optimplotfval);
[phat, fval, exitflag,output] = fminsearchstep(scaling_optimal,@funregister,initializing_point,options,live_new,mask_new,v3,method)
%find error and the image registered
[error, image2_registered] = funregisterdisplay(phat,live_new,mask_new, method);
image2_subtracted= live_new- image2_registered;
images = { live_new, image2_registered,image_subtracted_without_reg, image2_subtracted}
labels = {
        "Live", "Mask registered with " + method+ " inital p: " + initializing_point(1)+","+initializing_point(2)+",scaling:"+scaling_optimal+" error: " + error,"Subtracted image without registration",  "Subtracted image with registration"+scaling_optimal};
showImages(images,labels,'Q5_Resulting_Images',"A- Images with scaling and tolerance "+ scaling_optimal +" inital p: " + initializing_point(1)+","+initializing_point(2)+" with " + method + "step p: "+ phat(1)+","+phat(2), 1,4);
writeToVideo(image2_subtracted , v3,"Final Optimal One p: ("+ phat(1)+","+phat(2)+")"  )
close(v3);
disp(" iteration number is " + num2str(output.iterations))
%%
figure
image2_subtracted2 = 100 - image2_subtracted
imshow(image2_subtracted2)
dsa = imfuse(live_new,imcomplement(image2_registered),'blend','Scaling','joint');
imtool(dsa,'DisplayRange',[])

function for_table =part4b(contrast1_new, contrast2_new, scaling, p, similarity_funct)
    for_table = {}
    phats = zeros(length(scaling),2);
    errors = [];
    all_images = {}
    all_labels = {}
    %We need to loop over for each scaling value
    for i=1:length(scaling)
        images = {};
        lables = {};
        video_name = "Q4optimize_scaling_"+similarity_funct+"_"+p(1)+"_"+p(2)+"_"+scaling(i)+".avi"
        v3 = VideoWriter('Q4_Resulting_Videos/'+video_name)
        v3.FrameRate = 1;
        open(v3)
        %subtracted image without registration
        image_subtracted_without_reg = contrast1_new - contrast2_new;
        writeToVideo(image_subtracted_without_reg,v3,"Initial subtraction without the registration");
        %Create options
        % define options e.g. (doc optimset for details)
        options = optimset('TolX',1e-5,'TolFun',1e-5,'PlotFcns', @optimplotfval);
        [phat, fval, exitflag,output] = fminsearch(@funregisterscale,p,options,contrast1_new,contrast2_new,v3,similarity_funct,scaling(i))
        phats(i,1) = phat(1);
        phats(i,2) = phat(2);
         %find error and the image registered
        [error, image2_registered] = funregisterdisplay(phat,contrast1_new,contrast2_new, similarity_funct);
        errors(i) = error;
        image2_subtracted= contrast1_new - image2_registered;
        images = { contrast1_new,  image2_registered,image_subtracted_without_reg, image2_subtracted}
        labels = {
        "Contrast 1", "Contrast 2 registered with " + similarity_funct+ " inital p: " + p(1)+","+p(2)+",tol:"+scaling(i)+" error: " + error,"Subtracted image without registration",  "Subtracted image with registration"+scaling(i)};
        writeToVideo(image2_subtracted , v3,"Final Optimal One p: ("+ phat(1)+","+phat(2)+")"  )
        close(v3);
        disp("At scaling " + num2str(scaling(i))+ " iteration number is " + num2str(output.iterations))
        for_table=[for_table; {similarity_funct, p,scaling(i),output.iterations,phat,error}]
        all_images = [all_images, images]
        all_labels = [all_labels,labels]
    end
      showImages(all_images,all_labels,'Q4_Resulting_Images',"B- Images with different scaling  inital p: " + p(1)+","+p(2)+" with " + similarity_funct + "step p: "+ phat(1)+","+phat(2), 4,4);
end

function for_table = part4a(contrast1_new, contrast2_new, tolerances, p, similarity_funct)
    for_table = {}
    phats = zeros(length(tolerances),2);
    all_images = {}
    all_labels = {}
    errors = [];
    %We need to loop over for each tolerance value
    for i=1:length(tolerances)
        images = {};
        lables = {};
        video_name = "Q5optimize_"+similarity_funct+"_"+p(1)+"_"+p(2)+"_"+tolerances(i)+".avi"
        v3 = VideoWriter('Q5_Resulting_Videos/'+video_name)
        v3.FrameRate = 1;
        open(v3)
        %subtracted image without registration
        image_subtracted_without_reg = contrast1_new - contrast2_new;
        writeToVideo(image_subtracted_without_reg,v3,"Initial subtraction without the registration");
        %Create options
        % define options e.g. (doc optimset for details)
        options = optimset('TolX',tolerances(i),'TolFun',tolerances(i),'PlotFcns', @optimplotfval);
        [phat, fval, exitflag,output] = fminsearch(@funregister,p,options,contrast1_new,contrast2_new,v3,similarity_funct)
        phats(i,1) = phat(1);
        phats(i,2) = phat(2);
        %find error and the image registered
        [error, image2_registered] = funregisterdisplay(phat,contrast1_new,contrast2_new, similarity_funct);
        errors(i) = error;
        image2_subtracted= contrast1_new - image2_registered;
        images = { contrast1_new,  image2_registered,image_subtracted_without_reg, image2_subtracted}
        labels = {
        "Live", "Mask registered with " + similarity_funct+ " inital p: " + p(1)+","+p(2)+",tol:"+tolerances(i)+" error: " + error,"Subtracted image without registration",  "Subtracted image with registration"+tolerances(i)};
        writeToVideo(image2_subtracted , v3,"Final Optimal One p: ("+ phat(1)+","+phat(2)+")"  )
        close(v3);
        disp("At tolfunc " + num2str(tolerances(i))+ " iteration number is " + num2str(output.iterations))
        for_table=[for_table; {similarity_funct, p,tolerances(i),output.iterations,phat,error}]
        all_images = [all_images, images]
        all_labels = [all_labels,labels]
    end
      showImages(all_images,all_labels,'Q5_Resulting_Images',"A- Images with different tolerance inital p: " + p(1)+","+p(2)+" with " + similarity_funct, length(tolerances),4);
end

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
    
    % 3a- Used your function, myNCC
    if similarity_cost_function_name == "myNCC"
    % Compute myNCC after registration
    similarity_registered = myNCC(image1,image2_registered)
    error = 1-similarity_registered
    subtracted_img = image1 - image2_registered;
    writeToVideo(subtracted_img,v,"("+p(1)+","+p(2)+") error: "+ error); %for part 3d
    % 3b- Use x and y translation only (no rotation, no scaling, no shearing). You will need to
    % evaluate a similarity cost function (that is, SSE)
    elseif similarity_cost_function_name == "SSE"
        error = sumsqr(image1-image2_registered)
        subtracted_img = image1 - image2_registered;
        writeToVideo(subtracted_img,v,"("+p(1)+","+p(2)+") error: "+error); %for part 3d
    else
        fprintf('You need to give the name of the similarity cost function');
    end
end
function [error] = funregisterscale(p,image1,image2, v, similarity_cost_function_name,scale)
    disp("p (" + p(1) + "," +p(2) + ")")
    if p(1) <= 0.1
        x = p(1)*100*scale;
    else 
         x = p(1)*scale;
    end
    if p(2) <= 0.1
        y = p(2)*100*scale;
    else 
         y = p(2)*scale;
    end
    %defines transformations - translation matrix (I applied 2d affine func)
    trans = affine2d([1 0 0; 0 1 0; x y 1]);
    
    %initializes imref2d object
    outView = imref2d([size(image1),size(image1)]);
    
    % To see the translation, both the filled values black and white is defined
    % with imwarp, cubic interpolation is used
    image2_registered = imwarp(image2,trans,'cubic','OutputView',outView,'FillValues',255);
    
    % 3a- Used your function, myNCC
    if similarity_cost_function_name == "myNCC"
    % Compute myNCC after registration
    similarity_registered = myNCC(image1,image2_registered)
    error = 1-similarity_registered
    subtracted_img = image1 - image2_registered;
    writeToVideo(subtracted_img,v,"("+p(1)+","+p(2)+") error: "+ error); %for part 3d
    % 3b- Use x and y translation only (no rotation, no scaling, no shearing). You will need to
    % evaluate a similarity cost function (that is, SSE)
    elseif similarity_cost_function_name == "SSE"
        error = sumsqr(image1-image2_registered)
        subtracted_img = image1 - image2_registered;
        writeToVideo(subtracted_img,v,"("+p(1)+","+p(2)+") error: "+error); %for part 3d
    else
        fprintf('You need to give the name of the similarity cost function');
    end
end
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
    % 3b- Use x and y translation only (no rotation, no scaling, no shearing). You will need to
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

% 3d- At each step in the optimization algorithm (when each new Tx and Ty is computed),
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





