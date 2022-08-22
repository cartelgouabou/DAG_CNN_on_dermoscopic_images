%add halo artificially
%Arthur C Foahom.
%ratio= [0.05;0.15;0.45]
clear all
close all
%C:\Users\arthu\Desktop\melVSnev\NORM\TEST\MAL
img_folders = dir('C:\Users\arthu\Desktop\melVSnev\NORM\TEST\MAL\*.jpg');
N= length( img_folders );

for i= 1: N
    img_folders(i).name;
    filename=['C:\Users\arthu\Desktop\melVSnev\NORM\TEST\MAL\',img_folders(i).name];
    I=imread(filename);
    imgTran = addHalo(I,0.45);
    FileName = strcat(img_folders(i).name(1:end-4),'_grad_45.jpg');    
    imwrite(imgTran,['C:\Users\arthu\Desktop\melVSnev\NORM\TEST_45\MAL\',FileName]); 
end ;