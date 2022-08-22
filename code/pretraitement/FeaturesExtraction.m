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
%train
dataTrainBEKDir = fullfile('D:','PROJET','ISBI_ARTICLE','base','STEP2','STRONG','BEK');
dataTrainMELDir = fullfile('D:','PROJET','ISBI_ARTICLE','base','STEP2','STRONG','MEL');
dataTrainNEVDir = fullfile('D:','PROJET','ISBI_ARTICLE','base','STEP2','STRONG','NEV');
%test
dataTestDir = fullfile('D:','PROJET','ISBI_ARTICLE','base','BASE_MULTI','TEST');
%destination
pathDest = 'D:\PROJET\MICCAI_REVIS\base\STEP2\STRONG\';
%Récupération des images de la base de données
dataTrainBEK = datastore(dataTrainBEKDir);
dataTrainMEL = datastore(dataTrainMELDir);
dataTrainNEV = datastore(dataTrainNEVDir);
%%Jeu d'entrainement

%%test pour 1 image
path=char(dataTrainBEK.Files(1));
img=imread(path);
path (46:end); %find position debut nom du fichier
pos0=46;
filename = path(pos0:end);

options = configurationDescripteur(); % données de configuration du descripteurs
%%test LBP features pour 1 image
%imgResize=img;
%imgResize=ImageResize(img,800,600);

lbpFeatures = mlhmslbp_spyr(img , options{:});
lbpFeatureSize = length(lbpFeatures);


%%extraction des features dans toutes la base
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
    img=imread(path);
    filename=path(pos0:end);
    dataTrainBEKFilename{i+1,1}= path;
%     dataTrainBEKLabel(i+1,1)=0;
    lbpFeatures = mlhmslbp_spyr(img,options{:}); 
    F = lbpFeatures';
    dataTrainBEKFeatures(i+1, :) = F;
end

%extraction features mel
for i = 1:num1
    path=char(dataTrainMEL.Files(i));
    img=imread(path);
    filename=path(pos0:end);
    dataTrainMELFilename{i+1,1}= path;
%     dataTrainMELLabel(i+1,1)=1;
    lbpFeatures = mlhmslbp_spyr(img,options{:}); 
    F = lbpFeatures';
    dataTrainMELFeatures(i+1, :) = F;
end
%extraction features NEV
for i = 1:num2
    path=char(dataTrainNEV.Files(i));
    img=imread(path);
    filename=path(pos0:end);
    dataTrainNEVFilename{i+1,1}= path;
%     dataTrainNEVLabel(i+1,1)=2;
    lbpFeatures = mlhmslbp_spyr(img,options{:}); 
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

%%test pour 1 image
path=char(testpath.filepath(1));
img=imread(path);
path(49:end); %find position debut nom du fichier
pos0=49;
filename=path(pos0:end);

options = configurationDescripteur(); % données de configuration du descripteurs
%%test LBP features pour 1 image
%imgResize=img;
%imgResize=ImageResize(img,800,600);

lbpFeatures = mlhmslbp_spyr(img , options{:});
lbpFeatureSize = length(lbpFeatures);


%%extraction des features dans toutes la base
%Lesion Benigne data
num = numel(testpath.filepath);              %décompte nombre d'image dans la base de données
dataTestFeatures = zeros(num, lbpFeatureSize, 'single');     
dataTestFilename=cell(num,1);
%en-tête 
for i = 1:lbpFeatureSize
    dataTestFeatures(1,i) = i;   
end

%extraction features test
for i = 1:num
    path=char(testpath.filepath(i));
    img=imread(path);
    filename=path(pos0:end);
    dataTestFilename{i,1}= path;
    %dataTestLabel(i+1,1)=0;
    lbpFeatures = mlhmslbp_spyr(img,options{:}); 
    F = lbpFeatures';
    dataTestFeatures(i+1, :) = F;
end

csvwrite('dataTestFeatures.csv',dataTestFeatures);
dataFilename=cell2table(dataTestFilename);
writetable(dataFilename,'dataTestFilename.csv');
