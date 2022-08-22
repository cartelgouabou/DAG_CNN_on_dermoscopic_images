clear all
close all

img_folders = dir('D:\PROJET\DERMA_ARTICLE\base\HALO\2018\*.jpg');
 
 %R=I(:,:,2);
 %G=I(:,:,2);
 N= length( img_folders );
 for i= 1: N
  
    img_folders(i).name
    filename=['D:\PROJET\DERMA_ARTICLE\base\HALO\2018\',img_folders(i).name];
    I=imread(filename);
    figure(1);
    imshow(I);
    Ipt = double(I(:,:,1))./(double(I(:,:,1)+I(:,:,2))); %Pseudo-teinte
    figure(2);
    imshow(Ipt);
    
    Ipt=ipt(I); 
    figure(3);
    imshow(Ipt);
    %figure(3);
    %surf(Ipt);
    %figure(4);
    %imshow(100*Ipt,[0 255]);
    %figure(5);
    %imhist(Ipt);
    s = graythresh(Ipt) ; 
    IPTb = Ipt>0.6;
    %figure(6);
    %imshow(IPTb);
    figure(1);
    %IPTb= avoidCenter(IPTb,250);  %220
    imshow(IPTb);
    img_tran=I;
    img_tran(:,:,1)=img_tran(:,:,1).*uint8(IPTb);
    img_tran(:,:,2)=img_tran(:,:,2).*uint8(IPTb);
    img_tran(:,:,3)=img_tran(:,:,3).*uint8(IPTb);
    %figure(i);
    imshow(img_tran);
    %figure(7);
    %imhist(rgb2gray(I));
    %figure(8);
    %imhist(rgb2gray(img_tran));
    FileName = strcat(img_folders(i).name(1:end-4),'_pt','.jpg');    
    imwrite(IPTb,['D:\PROJET\DERMA_ARTICLE\base\HALO\2018\',FileName]); 
end ;

