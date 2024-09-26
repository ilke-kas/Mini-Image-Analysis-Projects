clear; clc; close all;
% Read Contrast1_new.tif
live_new = imread("live_new.tif");
% Read Contrast2_new.tif -- is a translated version of Contrast1-new by 1.8
% pixels and 2.1 pixels in the x and y directions, respectively
mask_new = imread("mask_new.tif");
live_new = im2gray(live_new)
mask_new = im2gray(mask_new)

moving = imhistmatch(mask_new,live_new);
iteration_arrays = [ [5000 400 200], [5000 400 200],[50 20 5], [50 20 5]]
AFS = [30 3 30 3]
%Before registration
before_registration = imfuse(live_new,imcomplement(mask_new),'blend','Scaling','joint');
imtool(before_registration ,'DisplayRange',[])
for i=1:length(AFS)
    [D,movingReg] = imregdemons(mask_new,live_new,iteration_arrays(i),...
    'AccumulatedFieldSmoothing',AFS(i),'PyramidLevels', 3,'DisplayWaitbar',true);
    dsa = imfuse(live_new,imcomplement(movingReg),'blend','Scaling','joint');
    imtool(dsa,'DisplayRange',[])

    figure
    [x, y] = meshgrid(1:size(D,2), 1:size(D,1));
    quiver(x, y, D(:,:,2), D(:,:,1));
    axis tight;
    title('Displacement Field (Quiver Plot)');
end






