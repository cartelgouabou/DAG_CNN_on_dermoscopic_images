%RANDOMLY SPLIT DATABASE
clear variables
clc;
close all;
pathSour = 'D:\PROJET\ISBI_ARTICLE\base\BASE_MULTI\TRAIN_VALID\'
pathDest = 'D:\PROJET\ISBI_ARTICLE\base\STEP2\STRONG\'
path=char(data.filename(2))
classname=path(1:3)
name=path(5:end)
path_img=fullfile(pathSour,classname,name)
img=imread(path_img);
num=height(data)
%Train/Test/Valid
for i = 2:num
    path=char(data.filename(i));
    classname=path(1:3);
    name=path(5:end);
    path_img=fullfile(pathSour,classname,name);
    img=imread(path_img);
    fullFileName = fullfile(pathDest,classname,name);
    imwrite(img,fullFileName);
end