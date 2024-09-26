clear; clc; close all;
% Read Contrast1_new.tif
contrast1_new = imread("Contrast1_new.tif");
% Read Contrast2_new.tif -- is a translated version of Contrast1-new by 1.8
% pixels and 2.1 pixels in the x and y directions, respectively
contrast2_new = imread("Contrast2_new.tif");

%% a- Calculate myNCC with the two input images
 similarity = myNCC(contrast1_new, contrast2_new);
 fprintf('Normalized Cross Correlation Similarity Between Two images: %.4f \n', similarity);
 images_a = {contrast1_new, contrast2_new}
 labels_a = {'Contrast1', 'Contrast2'}
 showImages(images_a,labels_a,'Q2_Resulting_Images',"A-Cross Correlation Similarity Between Two images "+ similarity, 2,1);

%% b- Translate Contrast2-new.tif by the required amounts so that the images are registered. -QUANTITATIVE
% %(This will require image interpolation,) Calculate myNCC. Report the value(s) obtained
% Define translation amounts
x = 1.8;
y = 2.1;
%defines transformations - translation matrix (I applied 2d affine func)
trans = affine2d([1 0 0; 0 1 0; x y 1]);

%initializes imref2d object
outView = imref2d([size(contrast1_new),size(contrast1_new)]);

% To see the traslation, both the filled values black and white is defined
% with imwarp, cubic interpolation is used
contrast2_registered_white = imwarp(contrast2_new,trans,'cubic','OutputView',outView,'FillValues',255)
contrast2_registered_black = imwarp(contrast2_new,trans,'cubic','OutputView',outView)

% Compute myNCC after registration
similarity_registered = myNCC(contrast1_new,contrast2_registered_white );

%display result
images_b = {contrast1_new, contrast2_new, contrast2_registered_black, contrast2_registered_white}
labels_b = {'Contrast1', 'Contrast2', 'Contrast2 Registered Filled Values Black','Contrast2 Registered Filled Values White'}
showImages(images_b,labels_b,'Q2_Resulting_Images',"B-Cross Correlation Similarity After Registration (White Filled Registered and Contrast1) "+ similarity_registered, 2,2);

%% c- Create a subtracted image to ensure that the images are registered. Display your final
% subtracted image.- QUALITATIVE
subtracted_image = contrast1_new - contrast2_registered_white
images_b = {contrast1_new, contrast2_new, contrast2_registered_black, contrast2_registered_white,subtracted_image}
labels_b = {'Contrast1', 'Contrast2', 'Contrast2 Registered Filled Values Black','Contrast2 Registered Filled Values White', 'Subtracted Image-Qualitative Evaluation'}
showImages(images_b,labels_b,'Q2_Resulting_Images',"C-Qualitative Evaluation", 3,2);


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

