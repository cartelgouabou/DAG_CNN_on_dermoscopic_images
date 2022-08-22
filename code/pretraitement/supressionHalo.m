[hau,lar,~]=size(img);
m1=mean(img);
m2=mean(img,2);


for i=1:lar
    cnt1=0;
    cnt2=0;
    cnt3=0;
    cnt4=0;
    
    NOM8DE LA VARAIABLE
    for j=1:hau
        t=img(j,i);
        if t<m1(i) && t<m2(j)  && cnt3==0
            cnt1=cnt1+1;
        else
            cnt3=cnt3+1;
        end
        k=img(hau-j+1,i);
        if k<m1(i) && cnt4==0 && k<m2(hau+1-j)  
            cnt2=cnt2+1;
        else
            cnt4=cnt4+1;
        end
     end
    M(1,i)=cnt1;
    M(2,i)=cnt2;
    
end
imwrite(imgTreat,fullFileName);

img2=img;

for k=1:lar
    c1=M(1,k);
    c2=M(2,k);
    img2(1:c1,k)= 255;
    img2((hau-c2+1):hau,k)= 255;
end

figure(1);imshow(img);
figure(2);imshow(img2);

