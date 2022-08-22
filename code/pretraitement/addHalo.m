%Add Halo artificialy
%Arthur C Foahom.

function[OUT] = addHalo(I,ratio)
    img_tran=I;
    img_mask=rgb2gray(I); 
    [hau,lar,dim]=size(I);
    r=ratio;
    if r>1 | dim<3
        OUT = img; disp ('Error, ratio must be comprise between[0-1] and the image of dim 3');
    else
        Oi=lar/2; Oj=hau/2;
        R=sqrt((1/pi)*(1-r)*(lar*hau));
        for i=1:lar
            for j=1:hau
                D = sqrt((i-Oi)^2 + (j-Oj)^2);
                if D<R
                    img_mask(j,i)=1;
                else
                    img_mask(j,i)=0;
                end
            end
        end
    end
    
    img_tran(:,:,1)=I(:,:,1).*uint8(img_mask);
    img_tran(:,:,2)=I(:,:,2).*uint8(img_mask);
    img_tran(:,:,3)=I(:,:,3).*uint8(img_mask);
    OUT=img_tran;
end
