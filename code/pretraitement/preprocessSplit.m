%RANDOMLY SPLIT DATABASE
clear variables
clc;
close all;

pathSource = 'D:\PROJET\REVUE_VISUAL_SENSOR\base\ISIC2018\'
pathTest = 'D:\PROJET\REVUE_VISUAL_SENSOR\base\TEST\'
pathTrain = 'D:\PROJET\REVUE_VISUAL_SENSOR\base\TRAIN\'
pathValid = 'D:\PROJET\REVUE_VISUAL_SENSOR\base\VALID\'
dataSource =  fullfile('D:','PROJET','REVUE_VISUAL_SENSOR','base','ISIC2018','MEL');
data = datastore(dataSource);
[num ~]=size(data.Files);
[train_idx,test_idx,valid_idx] = train_test_valid_split(num);
path=char(data.Files(1));
img=imread(path);
pos=49;
filename=path(pos:end);

%Train/Test/Valid
for i = 1:3
    if i==1
        temp='MEL';
    elseif i==2
        temp='NEV';
    elseif i==3
        temp='UNK';
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
    dataDir =  fullfile('D:','PROJET','REVUE_VISUAL_SENSOR','base','ISIC2018',temp);
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


