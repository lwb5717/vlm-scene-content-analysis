function [busy, ratio]=busyness(image, thr);

% calculates the busyness descriptor of an rgb image (image) from its greyscale version
% uses sobel edge detection with threshold (thr) - 
% typical thr : 0.01 (high sensitivity),  0.05 (mid sensitivity), 0.1 (low
% sensitivity)
% returns busyness (busy) as a percentage of the busy / non-busy areas
% returns ration (ratio) of busy / non-busy areas


im=rgb2gray(image);

 % use Gaussian blur with sd = 1 to slightly blur the greyscale image i to remove noise
 i= imgaussfilt(im,1); 
 
% adjusts image to range from 0-1 for filtering
 DI=imadjust(i, [], [0 1]);
 
 % apples sobel filter with threshold "thr"
 BWs=edge(DI, 'sobel', (graythresh(DI) * 0.1));
 %figure, imshow(BWs), title('binary gradient mask');
 
 %applies dilation
 se90=strel('line', 3, 90);
 se0=strel('line', 3, 0);
 BWsdil=imdilate(BWs, [se90 se0]);
 %figure, imshow(BWsdil), title('dilated edge mask');
 
 %fills gaps
 BWdfill=imfill(BWsdil, 'holes');
 %figure, imshow(BWdfill); title('binary image with filled holes');
 
 %applies erosion, final image is “BWfinal”
 seD=strel('diamond', 1);
 BWfinal=imerode(BWdfill, seD);
 BWfinal=imerode(BWfinal,seD);
 %figure, imshow(BWfinal), title('segmented image')

 %thresholded image size in number of pixels -> BWtotal
 BWsize=size(BWfinal);
 BWtotal=(BWsize(:,1)).*(BWsize(:,2));

 %counts the white number of pixels in the thresholded image (busy area)

 White=find(BWfinal==1); 
 Whitesize=size(White,1);
 
 %takes ratio of the white to the total number of pixels in the thresholded image (ratio of busy areas to total image area)
 ratio=Whitesize./BWtotal;
 
 %calculates image busyness as a percentage
 busy=ratio.*100;
