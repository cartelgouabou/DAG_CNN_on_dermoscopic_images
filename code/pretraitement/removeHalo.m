[hau,lar,~]=size(img);
im=rgb2gray(img);
im_mask=im;
R=280;
for i=1:lar
    for j=1:hau
        D = sqrt((i-300)^2 + (j-225)^2);
        if D<R
            im_mask(j,i)=1;
        else
            im_mask(j,i)=0;
        end
    end
end
%im_mask=im2uint8(im_mask);
img_tran=img;
for k=1:3
    for i=1:lar
        for j=1:hau
            if im_mask(j,i)==0
                img_tran(j,i,k)=255;
            end
        end
    end
end

figure(1);imshow(img);
figure(2);imshow(im_mask);
figure(3);imshow(img_tran);
imwrite(img_tran,'ISIC_0032524_sh.jpg')
