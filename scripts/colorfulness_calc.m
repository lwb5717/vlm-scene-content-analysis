function[CF] = colorfulness_calc(I)
I = im2double(I);
rg = I(:,:,1)-I(:,:,2);
yb = 0.5*(I(:,:,1)+I(:,:,2))-I(:,:,3);

CF = sqrt(var(var(rg)).^2+var(var(yb)).^2)+0.3*sqrt(mean2(rg).^2+mean2(yb).^2);
end