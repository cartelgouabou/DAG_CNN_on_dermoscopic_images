%RANDOMLY SPLIT DATABASE
clear variables
clc;
close all;

pathSource = 'D:\PROJET\BASE\ISIC2017\ISIC2017_Train\'
pathTest = 'D:\PROJET\ISBI_ARTICLE\base\BASE_MULTI\TEST_ISIC_2017\'

dataSource =  fullfile('D:','PROJET','BASE','ISIC2017','ISIC2017_Train','BEK');
data = datastore(dataSource);
[num ~]=size(data.Files);
% [train_idx,test_idx,valid_idx] = train_test_valid_split(num);
path=char(data.Files(1));
img=imread(path);
pos=44;
filename=path(pos:end);
pathSour = 'D:\PROJET\MICCAI_REVIS\base\BASE_MULTI\TRAIN_VALID\'
pathDest = 'D:\PROJET\MICCAI_REVIS\base\STEP2_ARTHUR\STRONG\'
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
%Train/Test/Valid
for i = 1:3
    if i==1
        temp='BEK';
    elseif i==2
        temp='MEL';
    elseif i==3
        temp='NEV';
    elseif i==4
        temp='NEV';
    elseif i==5
        temp='SCC';
    elseif i==6
        temp='ACK';
    elseif i==7
        temp='VAL';
    else 
        temp='DEF';
    end
    dataDir =  fullfile('D:','PROJET','MICCAI_REVIS','base','isic2018_O',temp);
    data = datastore(dataDir);
    num = numel(data.Files); 
    [train_idx,test_idx,valid_idx] = train_test_valid_split(num);
    for j = train_idx
        path=char(data.Files(j));
        img=imread(path);
        imgTreat=img;
        imgTreat = cropCenterISIC19(img);
        imgTreat = colorConstancy(imgTreat, 'gray world seg',2);
        filename=path(pos:end);
        fullFileName = fullfile(pathTrain,temp,filename);
        imwrite(imgTreat,fullFileName);
    end
    for k = test_idx
        path=char(data.Files(k));
        img=imread(path);
        imgTreat=img;
        imgTreat = cropCenterISIC19(img);
        imgTreat = colorConstancy(imgTreat, 'gray world seg',2);
        filename=path(pos:end);
        fullFileName = fullfile(pathTest,temp,filename);
        imwrite(imgTreat,fullFileName);
    end
    for l = valid_idx
        path=char(data.Files(l));
        img=imread(path);
        imgTreat=img;
        imgTreat = cropCenterISIC19(img);
        imgTreat = colorConstancy(imgTreat, 'gray world seg',2);
        filename=path(pos:end);
        fullFileName = fullfile(pathValid,temp,filename);
        imwrite(imgTreat,fullFileName);
    end
end