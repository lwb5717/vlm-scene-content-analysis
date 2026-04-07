function [m, sa, sb]=CIElabC(i);
% calculates CIElab colourfulnessfrom ab channels of an RGB image i (m1)
% also a, b channel standard deviations (sa, sb)
% sophie triantaphillidou, july2004

ilab=rgb2lab(i);
l=ilab(:,:,1);
a=ilab(:,:,2);
b=ilab(:,:,3);

sa=std2(a);
sb=std2(b);

m_array=sqrt(a.^2+b.^2);

m=mean2(m_array);




