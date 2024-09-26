%% Question 1- Write a function
% "myNCC.m" which computes normalized cross correlation between two input images.
% You will call the function as follows:
% R = mySumAbs(image1, image2)
% Here R is a measure of how much similarity there is between the two images (R=0 à perfectly
% similar).
function R = myNCC(image1,image2)
    % I used the Normalised Cross Corelation Formula given in Similarity Measures pdf
    %Find the mean values of the images in the overla region
    img1_mean = mean(image1, 'all'); % we used 'all' since in here, we return the one mean value of all elements of img1 and img2
    img2_mean = mean(image2,'all');
    %Find the difference between images and their mean
    img1_diff = image1-img1_mean;
    img2_diff = image2-img2_mean;
    numerator = sum(img1_diff .* img2_diff, 'all');
    denominator = sqrt(sumsqr(img1_diff)) * sqrt(sumsqr(img2_diff));
    R = numerator/denominator;
end