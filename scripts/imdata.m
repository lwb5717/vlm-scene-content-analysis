function [imagedata]=imdata()
%[imagedata]=imdata()
%
%The fuction reads in all tiff files from the current directory and
%calculates statistical and entropy descriptors from them: 
%image type (greyscale or colour), probability mass function, 
%mean, variance, median, mode and entropy.  
%Data is stored in the structure 'imagedata', also written to a text file with
%the same name.
%sophie triantaphillidou 2007

fid=fopen('imagedata.txt','a');
tiffiles=dir('*.tif');

S=size(tiffiles);
if S(1,1)==0 %checks to see if there are tiff files present
    error('there are no tiff files in the current directory')
end

names={tiffiles.name};
names=char(names);
s=size(names);
n=s(1,1);%identifies number of files

for a=1:n;
   I=imread(names(a,:));%reads in character string as filename 
   J=rgb2gray(I);
   %FILENAME
   imagedata(a).filename=names(a,:);%puts all filenames into filename field
   fprintf(fid,'name:%s\n',names(a,:));
   %TYPE
   b=size(I);
   if b(1,3)==1;
       imagedata(a).type='greyscale';
       fprintf(fid,'type:greyscale\n');
   else
       imagedata(a).type='colour';
        fprintf(fid,'type:colour\n');
   end 
   %PDF
   for g=1:256; %accesses each pixel value
       k=find(J==g);%finds each pixel value in the image
       h(g)=numel(k);%puts the number of times each pixel value occurs in the correct bin
   end
   h=h./(size(J,1)*size(J,2));
   imagedata(a).PDF=h;%normalises probability distribution
   z=num2str(h);
   fprintf(fid,'PDF:%s\n',z);    
      
   c=numel(J);
   e=zeros(1,c);
   for d=1:c;
       e(d)=J(d);%converts J to a row vector
   end
   %MEAN    
   m=mean(e);%calculated using row vector
   imagedata(a).mean=m;
   fprintf(fid,'mean:%g\n',m);
   %VARIANCE
   v=var(e);%calculated using row vector
   imagedata(a).variance=v;
   fprintf(fid,'variance:%g\n',v);
   %MEDIAN
   md=median(e);%calculated using row vector
   imagedata(a).median=md; 
   fprintf(fid,'median:%g\n',md);
   %MODE
   mode=find(h==(max(h)));
   imagedata(a).mode=mode;%mode is the pixel value at probability 1 as distribution is normalised
   fprintf(fid,'mode:%g\n',mode);
   %ENTROPY
   k=find(h~=0);  
   ent=-sum(h(k).*log2(h(k)));
   imagedata(a).entropy=ent;
   fprintf(fid,'entropy:%g\n\n',ent);
end
fclose(fid);