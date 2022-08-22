%split aléatoirement les index de la base
function [train_idx,test_idx,valid_idx] = train_test_split(num)
    alea_idx=randperm(num);
    train_idx = alea_idx(1:round(0.7*length(alea_idx)));
    test_idx = alea_idx((round(0.7*length(alea_idx))+1:(round(0.7*length(alea_idx))+ round(0.2*length(alea_idx)))));
    valid_idx = alea_idx((round(0.7*length(alea_idx))+ round(0.2*length(alea_idx)))+1:end);
end