%Crop and center
%Arthur C Foahom.
% 
function [OUT] = cropCenter(I,t) %
    
    img=I;
    [hau,lar,~]=size(img);
    d1=hau-t;
    d2=lar-t;
    p1=rem(d1,2);
    p2=rem(d2,2);
    if t>lar | t>hau
        OUT = img; disp ('Error, t must be smaller than the sizes of the images');
    else
        if d1==0 && d2==0
            imgCC=img;
        else
            if p1==0 && p2==0
                for i=1:3
                    imgCC(:,:,i)=img(1+d1/2:hau-d1/2,1+d2/2:lar-d2/2,i);
                end
            elseif p1~=0 && p2~=0
                for i=1:3
                    imgCC(:,:,i)=img((1+round(d1/2)-1):(hau-round(d1/2)),(1+round(d2/2)-1):(lar-round(d2/2)),i);
                end
            elseif p1~=0 
                for i=1:3
                    imgCC(:,:,i)=img((1+round(d1/2)-1):(hau-round(d1/2)),1+d2/2:lar-d2/2,i);
                end
            else  
                for i=1:3
                    imgCC(:,:,i)=img(1+d1/2:hau-d1/2,(1+round(d2/2)-1):(lar-round(d2/2)),i);
                end
            end
        end
        OUT = uint8(imgCC);
    end
end
    