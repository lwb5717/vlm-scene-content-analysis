function [chroma_m, hue_m]=ChromaHue(i);
% calculates CIELAB C* and h* (in degrees)
% and mean C*, mean h*
% sophie triantaphillidou 2007

ilab=rgb2lab(i);
l=ilab(:,:,1);
a=ilab(:,:,2);
b=ilab(:,:,3);

[x,y]=size(a);
chroma=zeros(x,y);
hue_rad=zeros(x,y);

chroma=sqrt(a.^2 + b.^2);
chroma_m=mean2(chroma);

i0=find(a==0 & b==0);
    hue_rad(i0)=0;

i01=find(a==0 & b>0);
    hue_rad(i01)=1.570796327;

i11=find(a==0 & b<0);
    hue_rad(i11)=+1.570796327+pi;

i=find(a ~= 0);
hue_rad(i)=atan(b(i) ./ a(i)); % hue angle in radians

hue=hue_rad.*(180/pi); % hue angle in degrees

aphn=find(a>0 & hue<0);
hue(aphn)=360+hue(aphn);
     
anhn=find(a<0);
hue(anhn)=180+hue(anhn);

hue_m=mean2(hue);
hue_sd=std2(hue);



 