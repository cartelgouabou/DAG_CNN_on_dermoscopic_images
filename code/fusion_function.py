#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:41:57 2022

@author: arthur
"""
import numpy as np
from scipy.stats import gmean

def ddag_fun(root_node,bekVSmel,bekVSnev,melVSnev):
    score_bek=[]
    score_mel=[]
    score_nev=[]
    predict_final=[]
    #bekVSmel # x>0.5 = 1 ; x<0.5 = 0
    #bekVSnev # x>0.5 = 2 ; x<0.5 = 0
    #melVSnev # x>0.5 = 2 ; x<0.5 = 1
    if root_node=='01':
        if (bekVSmel>0.5) & (melVSnev<0.5) : #not0 -- not2
                score_bek=(1-bekVSmel)/((1-bekVSmel)+(1-melVSnev)+melVSnev)
                score_mel=(1-melVSnev)/((1-bekVSmel)+(1-melVSnev)+melVSnev)
                score_nev=melVSnev/((1-bekVSmel)+(1-melVSnev)+melVSnev)
                predict_final=1
        elif (bekVSmel>0.5) & (melVSnev>0.5) : #not0 -- not2
                score_bek=(1-bekVSmel)/((1-bekVSmel)+(1-melVSnev)+melVSnev)
                score_mel=(1-melVSnev)/((1-bekVSmel)+(1-melVSnev)+melVSnev)
                score_nev=melVSnev/((1-bekVSmel)+(1-melVSnev)+melVSnev)
                predict_final=2
        elif (bekVSmel<0.5) & (bekVSnev>0.5) : #not1 -- not0
                score_bek=(1-bekVSnev)/((1-bekVSnev)+bekVSmel+bekVSnev)
                score_mel=bekVSmel/((1-bekVSmel)+bekVSmel+bekVSnev)
                score_nev=bekVSnev/((1-bekVSmel)+bekVSmel+bekVSnev)
                predict_final=2
        else  : #not1 -- not2
                score_bek=(1-bekVSnev)/((1-bekVSnev)+bekVSmel+bekVSnev)
                score_mel=bekVSmel/((1-bekVSmel)+bekVSmel+bekVSnev)
                score_nev=bekVSnev/((1-bekVSmel)+bekVSmel+bekVSnev)
                predict_final=0
    elif root_node=='02':
        if (bekVSnev>=0.5) & (melVSnev>=0.5) : #not0 -- not1
            score_bek=(1-bekVSnev)/((1-bekVSnev)+(1-melVSnev)+melVSnev)
            score_mel=(1-melVSnev)/((1-bekVSmel)+(1-melVSnev)+melVSnev)
            score_nev=melVSnev/((1-bekVSmel)+(1-melVSnev)+melVSnev)
            predict_final=2
        elif (bekVSnev>=0.5) & (melVSnev<=0.5) : #not0 -- not2
            score_bek=(1-bekVSnev)/((1-bekVSnev)+(1-melVSnev)+melVSnev)
            score_mel=(1-melVSnev)/((1-bekVSmel)+(1-melVSnev)+melVSnev)
            score_nev=melVSnev/((1-bekVSmel)+(1-melVSnev)+melVSnev)
            predict_final=1
        elif (bekVSnev<=0.5) & (bekVSmel>=0.5) : #not2 -- not0
            score_bek=(1-bekVSmel)/((1-bekVSmel)+bekVSmel+bekVSnev)
            score_mel=bekVSmel/((1-bekVSmel)+bekVSmel+bekVSnev)
            score_nev=bekVSnev/((1-bekVSmel)+bekVSmel+bekVSnev)
            predict_final=1
        else  : #not2 -- not1
            score_bek=(1-bekVSnev)/((1-bekVSnev)+bekVSmel+bekVSnev)
            score_mel=bekVSmel/((1-bekVSnev)+bekVSmel+bekVSnev)
            score_nev=bekVSnev/((1-bekVSnev)+bekVSmel+bekVSnev)
            predict_final=0
    elif root_node=='12':
        if (melVSnev>0.5) & (bekVSnev>0.5) : #not1 -- not0
            score_bek=(1-bekVSnev)/((1-bekVSnev)+(1-melVSnev)+bekVSnev)
            score_mel=(1-melVSnev)/((1-bekVSnev)+(1-melVSnev)+bekVSnev)
            score_nev=bekVSnev/((1-bekVSnev)+(1-melVSnev)+bekVSnev)
            predict_final=2
        elif (melVSnev>0.5) & (bekVSnev<0.5) : #not1 -- not2
            score_bek=(1-bekVSnev)/((1-bekVSnev)+(1-melVSnev)+bekVSnev)
            score_mel=(1-melVSnev)/((1-bekVSnev)+(1-melVSnev)+bekVSnev)
            score_nev=bekVSnev/((1-bekVSnev)+(1-melVSnev)+bekVSnev)
            predict_final=0
        elif (melVSnev<0.5) & (bekVSmel<0.5) : #not2 -- not1
            score_bek=(1-bekVSmel)/((1-bekVSmel)+bekVSmel+melVSnev)
            score_mel=bekVSmel/((1-bekVSmel)+bekVSmel+melVSnev)
            score_nev=melVSnev/((1-bekVSmel)+bekVSmel+melVSnev)
            predict_final=0
        else  : #not2 -- not0
            score_bek=(1-bekVSmel)/((1-bekVSmel)+bekVSmel+melVSnev)
            score_mel=bekVSmel/((1-bekVSmel)+bekVSmel+melVSnev)
            score_nev=melVSnev/((1-bekVSmel)+bekVSmel+melVSnev)
            predict_final=1
    else:
        print('specifie root node')
    return score_bek,score_mel,score_nev,predict_final





def avg_fun(bek_cnn1,bek_cnn2,bek_cnn3,mel_cnn1,mel_cnn2,mel_cnn3,nev_cnn1,nev_cnn2,nev_cnn3): #average strategie
    score_bek=[]
    score_mel=[]
    score_nev=[]
    predict_final=[]
    score_bek=np.mean([bek_cnn1,bek_cnn2,bek_cnn3])
    score_mel=np.mean([mel_cnn1,mel_cnn2,mel_cnn3])
    score_nev=np.mean([nev_cnn1,nev_cnn2,nev_cnn3])
    score_bek_N=score_bek/(score_bek+score_mel+score_nev)
    score_mel_N=score_mel/(score_bek+score_mel+score_nev)
    score_nev_N=score_nev/(score_bek+score_mel+score_nev)
    predict_final=np.argmax([score_bek_N,score_mel_N,score_nev_N])
    return score_bek_N,score_mel_N,score_nev_N,predict_final
def gmean_fun(bek_cnn1,bek_cnn2,bek_cnn3,mel_cnn1,mel_cnn2,mel_cnn3,nev_cnn1,nev_cnn2,nev_cnn3): #average strategie
    score_bek=[]
    score_mel=[]
    score_nev=[]
    predict_final=[]
    score_bek=gmean([bek_cnn1,bek_cnn2,bek_cnn3])
    score_mel=gmean([mel_cnn1,mel_cnn2,mel_cnn3])
    score_nev=gmean([nev_cnn1,nev_cnn2,nev_cnn3])
    score_bek_N=score_bek/(score_bek+score_mel+score_nev)
    score_mel_N=score_mel/(score_bek+score_mel+score_nev)
    score_nev_N=score_nev/(score_bek+score_mel+score_nev)
    predict_final=np.argmax([score_bek_N,score_mel_N,score_nev_N])
    return score_bek_N,score_mel_N,score_nev_N,predict_final
def max_conf_fun(bek_cnn1,bek_cnn2,bek_cnn3,mel_cnn1,mel_cnn2,mel_cnn3,nev_cnn1,nev_cnn2,nev_cnn3): #max confidence strategy
    score_bek=[]
    score_mel=[]
    score_nev=[]
    score_bek=np.max([bek_cnn1,bek_cnn2,bek_cnn3])
    score_mel=np.max([mel_cnn1,mel_cnn2,mel_cnn3])
    score_nev=np.max([nev_cnn1,nev_cnn2,nev_cnn3])
    score_bek_N=score_bek/(score_bek+score_mel+score_nev)
    score_mel_N=score_mel/(score_bek+score_mel+score_nev)
    score_nev_N=score_nev/(score_bek+score_mel+score_nev)
    predict_final=np.argmax([score_bek_N,score_mel_N,score_nev_N])
    return score_bek_N,score_mel_N,score_nev_N,predict_final
def prod_fun(bek_cnn1,bek_cnn2,bek_cnn3,mel_cnn1,mel_cnn2,mel_cnn3,nev_cnn1,nev_cnn2,nev_cnn3): #product of prob strategy
    score_bek=[]
    score_mel=[]
    score_nev=[]
    score_bek=bek_cnn1*bek_cnn2*bek_cnn3
    score_mel=mel_cnn1*mel_cnn2*mel_cnn3
    score_nev=nev_cnn1*nev_cnn2*nev_cnn3
    score_bek_N=score_bek/(score_bek+score_mel+score_nev)
    score_mel_N=score_mel/(score_bek+score_mel+score_nev)
    score_nev_N=score_nev/(score_bek+score_mel+score_nev)
    predict_final=np.argmax([score_bek_N,score_mel_N,score_nev_N])
    return score_bek_N,score_mel_N,score_nev_N,predict_final
def svm_fun(bek_cnn1,bek_cnn2,bek_cnn3,mel_cnn1,mel_cnn2,mel_cnn3,nev_cnn1,nev_cnn2,nev_cnn3): #simple voting strategy strategy
    bek_voter=[]
    mel_voter=[]
    nev_voter=[]
    bek=[]
    mel=[]
    nev=[]
    bek_voter=np.array([bek_cnn1,bek_cnn2,bek_cnn3])
    mel_voter=np.array([mel_cnn1,mel_cnn2,mel_cnn3])
    nev_voter=np.array([nev_cnn1,nev_cnn2,nev_cnn3])
    bek=len(bek_voter[bek_voter>=0.5])
    mel=len(mel_voter[mel_voter>=0.5])
    nev=len(nev_voter[nev_voter>=0.5])
    score_bek_N=bek/3
    score_mel_N=mel/3
    score_nev_N=nev/3
    predict_final=np.argmax([score_bek_N,score_mel_N,score_nev_N])
    return score_bek_N,score_mel_N,score_nev_N,predict_final