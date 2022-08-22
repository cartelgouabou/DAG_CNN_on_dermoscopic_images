# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:59:22 2021

@author: arthu
"""
Bl=0
Bll=0
Ml=0
Mll=0
Nl=0
Nll=0
Blr=0
Mlr=0
Nlr=0

Br=0
Brl=0
Mr=0
Mrl=0
Nr=0
Nrl=0
Brr=0
Mrr=0
Nrr=0
for i in range(len(test_set)):
    bekVSmel=result.bekVSmel[i] # x>0.5 = 1 ; x<0.5 = 0
    bekVSnev=result.bekVSnev[i] # x>0.5 = 2 ; x<0.5 = 0
    melVSnev=result.melVSnev[i] # x>0.5 = 2 ; x<0.5 = 1
    if (bekVSnev>=0.5): #l
        if y_test[i]==0:
                Bl+=1
        elif y_test[i]==1:
                Ml+=1
        else:
                Nl+=1
        if melVSnev>=0.5:#ll
            if y_test[i]==0:
                Bll+=1
            elif y_test[i]==1:
                Mll+=1
            else:
                Nll+=1
        else: #lr
            if y_test[i]==0:
                Blr+=1
            elif y_test[i]==1:
                Mlr+=1
            else:
                Nlr+=1
    else : #r
        if y_test[i]==0:
                Br+=1
        elif y_test[i]==1:
                Mr+=1
        else:
                Nr+=1
        if (bekVSmel>=0.5) : #rl
            if y_test[i]==0:
                Brl+=1
            elif y_test[i]==1:
                Mrl+=1
            else:
                Nrl+=1
        else: #rr
            if y_test[i]==0:
                Brr+=1
            elif y_test[i]==1:
                Mrr+=1
            else:
                Nrr+=1
            