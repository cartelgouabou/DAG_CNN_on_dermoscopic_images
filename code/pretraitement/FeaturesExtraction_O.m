% : Extract features strong image  (10/2020)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Script to compute features with direct approach 
%
% %% Modified Version
% Author: Arthur Foahom

clc;
clear all;
close all;
%Source
path_O = 'D:\PROJET\ISBI_ARTICLE\base\isic2018_O\'
%train
dataTrainBEKDir = fullfile('D:','PROJET','ISBI_ARTICLE','base','BASE_MULTI','TRAIN_VALID','BEK');
dataTrainMELDir = fullfile('D:','PROJET','ISBI_ARTICLE','base','BASE_MULTI','TRAIN_VALID','MEL');
dataTrainNEVDir = fullfile('D:','PROJET','ISBI_ARTICLE','base','BASE_MULTI','TRAIN_VALID','NEV');

%Récupération des images de la base de données
dataTrainBEK = datastore(dataTrainBEKDir);
dataTrainMEL = datastore(dataTrainMELDir);
dataTrainNEV = datastore(dataTrainNEVDir);

%%test pour 1 image 

path=char(dataTrainBEK.Files(1));
classname='BEK'
img=imread(path);
path (56:end); %find position debut nom du fichier
pos0=56;
filename = path(pos0:end);
path_img=fullfile(path_O,classname,filename);
img=imread(path_img);    
imgTreat = cropCenterISIC19(img);
options = configurationDescripteur(); % données de configuration du descripteurs
%%test LBP features pour 1 image
%imgResize=img;
%imgResize=ImageResize(img,800,600);

lbpFeatures = mlhmslbp_spyr(img , options{:});
lbpFeatureSize = length(lbpFeatures);

%Lesion Benigne data
num0 = numel(dataTrainBEK.Files);              %décompte nombre d'image dans la base de données
num1 = numel(dataTrainMEL.Files);
num2 = numel(dataTrainNEV.Files);
dataTrainBEKFeatures = zeros(num0+1, lbpFeatureSize, 'single');     
dataTrainMELFeatures = zeros(num1+1, lbpFeatureSize, 'single');     
dataTrainNEVFeatures = zeros(num2+1, lbpFeatureSize, 'single');     

%dataTrainBEKLabel = zeros(num0+1, 1, 'single');    
dataTrainBEKFilename=cell(num0,1);
dataTrainMELFilename=cell(num1,1);
dataTrainNEVFilename=cell(num2,1);
%dataTrainFilename{1,1}= 'Filename';
%en-tête 
for i = 1:lbpFeatureSize
    dataTrainBEKFeatures(1,i) = i;   
end
for i = 1:lbpFeatureSize
    dataTrainMELFeatures(1,i) = i;   
end
for i = 1:lbpFeatureSize
    dataTrainNEVFeatures(1,i) = i;   
end


%extraction features bek
for i = 1:num0
    path=char(dataTrainBEK.Files(i));
    classname='BEK';
    filename=path(pos0:end);
    path_img=fullfile(path_O,classname,filename);
    img=imread(path_img);    
    imgTreat = cropCenterISIC19(img);
    lbpFeatures = mlhmslbp_spyr(imgTreat , options{:});
    F = lbpFeatures';
    dataTrainBEKFeatures(i+1, :) = F;
end

%extraction features mel
for i = 1:num1
    path=char(dataTrainMEL.Files(i));
    classname='MEL';
    filename=path(pos0:end);
    path_img=fullfile(path_O,classname,filename);
    img=imread(path_img);    
    imgTreat = cropCenterISIC19(img);
    lbpFeatures = mlhmslbp_spyr(imgTreat , options{:});
    F = lbpFeatures';
    dataTrainMELFeatures(i+1, :) = F;
end
%extraction features NEV
for i = 1:num2
    path=char(dataTrainNEV.Files(i));
    classname='NEV';
    filename=path(pos0:end);
    path_img=fullfile(path_O,classname,filename);
    img=imread(path_img);    
    imgTreat = cropCenterISIC19(img);
    lbpFeatures = mlhmslbp_spyr(imgTreat , options{:});
    F = lbpFeatures';
    dataTrainNEVFeatures(i+1, :) = F;
end


csvwrite('dataTrainBEKFeatures.csv',dataTrainBEKFeatures);
dataFilename=cell2table(dataTrainBEKFilename);
writetable(dataFilename,'dataTrainBEKFilename.csv');
csvwrite('dataTrainMELFeatures.csv',dataTrainMELFeatures);
dataFilename=cell2table(dataTrainMELFilename);
writetable(dataFilename,'dataTrainMELFilename.csv');
csvwrite('dataTrainNEVFeatures.csv',dataTrainNEVFeatures);
dataFilename=cell2table(dataTrainNEVFilename);
writetable(dataFilename,'dataTrainNEVFilename.csv');

%%Jeu test
num = numel(testpath.filepath);
path=char(testpath.filepath(1));
classname=path(45:47)
filename=path(49:end)
path_img=fullfile(path_O,classname,filename)
img=imread(path_img);
imgTreat = cropCenterISIC19(img);
lbpFeatures = mlhmslbp_spyr(imgTreat , options{:});

dataTestFeatures = zeros(num, lbpFeatureSize, 'single');     
dataTestFilename=cell(num,1);
%en-tête 
for i = 1:lbpFeatureSize
    dataTestFeatures(1,i) = i;   
end

%extraction features test
for i = 1:num
    path=char(testpath.filepath(i));
    classname=path(45:47)
    filename=path(49:end)
    path_img=fullfile(path_O,classname,filename)
    img=imread(path_img);
    filename=path(49:end);
    dataTestFilename{i,1}= path;
    %dataTestLabel(i+1,1)=0;
    lbpFeatures = mlhmslbp_spyr(img,options{:}); 
    F = lbpFeatures';
    dataTestFeatures(i+1, :) = F;
end

csvwrite('dataTestFeatures.csv',dataTestFeatures);
dataFilename=cell2table(dataTestFilename);
writetable(dataFilename,'dataTestFilename.csv');
