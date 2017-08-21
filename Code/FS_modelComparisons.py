#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:26:40 2017

@author: Claire Mary Kelleher
MSc. Big Data Science 
Thesis Project: Interactional and Linguistic Analysis for Computationally Diagnosing Alzheimer’s Disease
"""

"""
Purpose of this script:
    
Feature selection of both Linguistic (Non-IF's) & Interactional features
Features correlation (Pearson) with diagnosis variable
Features correlation with each other 
Model build & Comparison (
        4 different cuts of data/target labels used
        4 different clf's used (DT, KNN, LR, NB)

"""
###############################################################################
#               Libraries imported
###############################################################################

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support
import statistics
from sklearn.preprocessing import LabelEncoder
import scipy
import warnings

###############################################################################
#               Results table for classifers created
###############################################################################

results_prf = []

###############################################################################
#               Import data as Pandas DataFrame
#               (Imported in separate pickles as too large to run at once)
###############################################################################
df1nif = pd.read_pickle('/Users/clairekelleher/Desktop/Thesis/Data/df0_to_2451.pkl')
df2nif = pd.read_pickle('/Users/clairekelleher/Desktop/Thesis/Data/df2470_to_eof.pkl')
df0if = pd.read_pickle('/Users/clairekelleher/Desktop/Thesis/Data/script1.pkl') ##Forgot to run on 1st script 
df1if = pd.read_pickle('/Users/clairekelleher/Desktop/Thesis/Data/IF_Start_to_1572.pkl')
df2if = pd.read_pickle('/Users/clairekelleher/Desktop/Thesis/Data/IF_1580_to_end.pkl')
df3_if_ansLen = pd.read_pickle('/Users/clairekelleher/Desktop/Thesis/Data/script_ansLen.pkl')
df_all_if = pd.read_pickle('/Users/clairekelleher/Desktop/Thesis/Data/new_IF_all.pkl')
df4_nif_parsed = pd.read_pickle('/Users/clairekelleher/Desktop/Thesis/Data/new_PARSE_all.pkl')

df1tnif = df1nif.transpose()
df2tnif = df2nif.transpose()
dfptnif = df4_nif_parsed.transpose()
dfptnif = dfptnif.drop('dx',1)
frames = [df1tnif,df2tnif]

dft_all_if = df_all_if.transpose()
df0tif = df0if.transpose()
df1tif = df1if.transpose()
df2tif = df2if.transpose()
df3tif = df3_if_ansLen.transpose().astype(object)


frames_nif = [df1tnif,df2tnif]
frames_if = [dft_all_if]

totalScriptsDF_nif = pd.concat(frames_nif)
feature_names_all_wdx = ["dx", "Adverb_Ct_Total","Avg_Word_Len","INV_Word_Ct","INVtoPAR_word_ratio","Keyword_cookie","Keyword_counter","Keyword_curtain","Keyword_sink","Keyword_stool","Keyword_window","Noun_Ct_Total","PAR_UTT_ct","PAR_Word_Ct","ProNoun_Ct","ProNoun_Ct","ProNoun_Noun_Ratio","Script_Letter_Ct","Script_Word_Ct","Subj_IU_Total","Total_Keywords_Mentioned","Verb_Ct_3sing","Verb_Ct_Base","Verb_Ct_Gerund","Verb_Ct_Past","Verb_Ct_Past_P","Verb_Ct_Past_T","Verb_Ct_Total","Verb_Ct_non3sing","cos_avg","wordsOnceOverTotal","wordsUsedOnce"] #Original NIF
totalScriptsDF_nif.columns = feature_names_all_wdx
totalScriptsDF_nif = totalScriptsDF_nif.loc[:,~totalScriptsDF_nif.columns.duplicated()]
totalScriptsDF_nif = pd.concat([totalScriptsDF_nif,dfptnif],axis=1)

totalScriptsDF_if = pd.concat(frames_if)
totalScriptsDF_if = totalScriptsDF_if.drop('dx', 1)
totalScriptsDF_if = pd.concat([totalScriptsDF_if,df3tif],axis=1)
totalScriptsDF_if = totalScriptsDF_if.loc[:,~totalScriptsDF_if.columns.duplicated()]


totalScriptsDF_both_ifnif = pd.concat([totalScriptsDF_nif, totalScriptsDF_if], axis=1)
totalScriptsDF_both_ifnif = totalScriptsDF_both_ifnif.loc[:,~totalScriptsDF_both_ifnif.columns.duplicated()]

###############################################################################
#               Feature Selection     
###############################################################################

###############################################################################
#                   NIF's  
###############################################################################
NIF_feats_1stRun_orig = list(totalScriptsDF_nif.columns) #Start off with 34 NIFs
i=0
while(i<len(NIF_feats_1stRun_orig)):
    if NIF_feats_1stRun_orig[i] == 'dx':
        del NIF_feats_1stRun_orig[i]
    if NIF_feats_1stRun_orig[i] == 'INV_Word_Ct':
        del NIF_feats_1stRun_orig[i]
    if NIF_feats_1stRun_orig[i] == 'INVtoPAR_word_ratio':
        del NIF_feats_1stRun_orig[i]
    if NIF_feats_1stRun_orig[i] == 'ROOT_to_FRAG': #Normalised versions remained
        del NIF_feats_1stRun_orig[i] 
    if NIF_feats_1stRun_orig[i] == 'VP_VBG':
        del NIF_feats_1stRun_orig[i] 
    if NIF_feats_1stRun_orig[i] == 'VP_to_AUX_VP':
        del NIF_feats_1stRun_orig[i]
    if NIF_feats_1stRun_orig[i] == 'NP_to_PRP':
        del NIF_feats_1stRun_orig[i]
    if NIF_feats_1stRun_orig[i] == 'NP_to_DT_NN':
        del NIF_feats_1stRun_orig[i] 
    if NIF_feats_1stRun_orig[i] == 'ADVP_to_RB':
        del NIF_feats_1stRun_orig[i]
    i=i+1
    
#Remove 11 variables based on corr plot:['Script_Letter_Ct','Script_Word_Ct','Total_Keywords_Mentioned','Verb_Ct_Past_T','Verb_Ct_Total','wordsUsedOnce']
NIF_feats_Post_FS = []
i=0
while(i<len(NIF_feats_1stRun_orig)):
    if NIF_feats_1stRun_orig[i] != 'Script_Letter_Ct' and NIF_feats_1stRun_orig[i] != 'Script_Word_Ct' and NIF_feats_1stRun_orig[i] != 'Total_Keywords_Mentioned' and NIF_feats_1stRun_orig[i] != 'Verb_Ct_Past_T' and NIF_feats_1stRun_orig[i] != 'Verb_Ct_Total' and NIF_feats_1stRun_orig[i] != 'wordsUsedOnce' and NIF_feats_1stRun_orig[i] != 'PAR_UTT_ct' and NIF_feats_1stRun_orig[i] != 'Verb_Ct_non3sing' and NIF_feats_1stRun_orig[i] != 'Verb_Ct_Base' and NIF_feats_1stRun_orig[i] != 'PAR_Word_Ct' and NIF_feats_1stRun_orig[i] != 'Verb_Ct_Past_P':
        NIF_feats_Post_FS += [NIF_feats_1stRun_orig[i]]
    i=i+1
""" Top NIFS - 23 Remaining
"""
###############################################################################
#                   IF's    
###############################################################################
IF_feats_orig = list(totalScriptsDF_if.columns)# count 37

"""
#4 'blockCStutter'
#5 'brokenWordStutter'
#13 'omitted'
#15 'phonologicalFragmentStutter'
#17 'postFillerPOS_Adjective'
#18 'postFillerPOS_Adverb'
#23 'reptitionIterStutter'
are constant.

Also remove:
'wholeWordRepetitionTypDys'
'ansWordLen2'
'phraseRepTypDys'
'prolongationStutter'
'multWholeWordRepetitionTypDys'
'parWordCount'

=> 37-13=  '''''24 orginally''''
"""

IF_feats_Pre_FS=[]
IF_feats_Post_FS = []
i=0
while(i<len(IF_feats_orig)):
    if (IF_feats_orig[i] != 'blockCStutter' 
    and IF_feats_orig[i] != 'brokenWordStutter' 
    and IF_feats_orig[i] != 'omitted' 
    and IF_feats_orig[i] != 'phonologicalFragmentStutter'
#    and IF_feats_orig[i] != 'postFillerPOS_Adjective'
#    and IF_feats_orig[i] != 'postFillerPOS_Adverb'
    and IF_feats_orig[i] != 'reptitionIterStutter'
    and IF_feats_orig[i] != 'wholeWordRepetitionTypDys'
    and IF_feats_orig[i] != 'phraseRepTypDys'
    and IF_feats_orig[i] != 'multWholeWordRepetitionTypDys'
    and IF_feats_orig[i] != 'ansWordLen2'
    and IF_feats_orig[i] != 'prolongationStutter'
    and IF_feats_orig[i] != 'parWordCount'):
        IF_feats_Pre_FS += [IF_feats_orig[i]]
    i=i+1

"""23 IF's post FS"""

i=0
while(i<len(IF_feats_Pre_FS)):
    if IF_feats_Pre_FS[i] != 'Interruption_Q':
        IF_feats_Post_FS += [IF_feats_Pre_FS[i]]
    i=i+1
    
###############################################################################
#               Merge NIF's and IF's   
###############################################################################

Top_NIF_feats = NIF_feats_Post_FS
Top_IF_feats = IF_feats_Post_FS

Top_NIF_IF_feats =  Top_NIF_feats + Top_IF_feats

totalScriptsDF_both_ifnif = totalScriptsDF_both_ifnif.loc[:,~totalScriptsDF_both_ifnif.columns.duplicated()]
#^count: 46 (23 NIFs + 23 IFs)

###############################################################################
#               Input features to DF and set predictor/target variables     
###############################################################################
x_featList_to_use_x = IF_feats_Pre_FS #######-------Insert Feat List from above here------######

feat_len = len(x_featList_to_use_x)
feature_names_pred = x_featList_to_use_x
target = 'dx' #######-------Target always diagnosis variable

totalScriptsDF = pd.concat(frames)
totalScriptsDF.columns = feature_names_all_wdx
totalScriptsDF = totalScriptsDF.loc[:,~totalScriptsDF.columns.duplicated()]

totalScriptsDF = totalScriptsDF_both_ifnif

target_df = totalScriptsDF[target]
pred_df = totalScriptsDF[x_featList_to_use_x]

###############################################################################
#               Encode binary variables to investiate correlation     
###############################################################################

totalScriptsDF_both_ifnif["Control_enc"]=np.where(totalScriptsDF_both_ifnif["dx"] == 'Control',1,0)
totalScriptsDF_both_ifnif["MCI_enc"]=np.where(totalScriptsDF_both_ifnif["dx"] == 'MCI',1,0)
totalScriptsDF_both_ifnif["Memory_enc"]=np.where(totalScriptsDF_both_ifnif["dx"] == 'Memory',1,0)
totalScriptsDF_both_ifnif["PossibleAD_enc"]=np.where(totalScriptsDF_both_ifnif["dx"] == 'PossibleAD',1,0)
totalScriptsDF_both_ifnif["ProbableAD_enc"]=np.where(totalScriptsDF_both_ifnif["dx"] == 'ProbableAD',1,0)
totalScriptsDF_both_ifnif["Vascular_enc"]=np.where(totalScriptsDF_both_ifnif["dx"] == 'Vascular',1,0)

lb_make = LabelEncoder()
totalScriptsDF_both_ifnif["dx_enc"] = lb_make.fit_transform(totalScriptsDF_both_ifnif["dx"])
totalScriptsDF_both_ifnif[["dx", "dx_enc"]].head(11)
totalScriptsDF_both_ifnif.groupby(totalScriptsDF_both_ifnif['dx_enc']).size()
totalScriptsDF_both_ifnif.groupby(totalScriptsDF_both_ifnif['dx']).size()

###############################################################################
#               Pearson Correlation between Prob AD Vs. Control                          
###############################################################################

corr_analysis_df = totalScriptsDF_both_ifnif[x_featList_to_use_x].join(target_df)

corr_analysis_df_mci = corr_analysis_df[corr_analysis_df["dx"] == "MCI"] 
corr_analysis_df_mem = corr_analysis_df[corr_analysis_df["dx"] == "Memory"]
corr_analysis_df_poss = corr_analysis_df[corr_analysis_df["dx"] == "PossibleAD"] 
corr_analysis_df_vas = corr_analysis_df[corr_analysis_df["dx"] == "Vascular"]
corr_analysis_other_dx = corr_analysis_df_mci + corr_analysis_df_mem + corr_analysis_df_poss + corr_analysis_df_vas
corr_analysis_df = corr_analysis_df.drop(corr_analysis_other_dx.index)
corr_analysis_df['dx_bi'] = np.where(corr_analysis_df["dx"] == 'Control',0,1)

for all in x_featList_to_use_x:
    a = corr_analysis_df['dx_bi']
    b = corr_analysis_df[all]
    pearcorr = scipy.stats.pearsonr(a, b)
    print(all)
    print(pearcorr) #returns: (Pearson’s correlation coefficient,2-tailed p-value)

###############################################################################
#               Feature prediction investigation
#               Select K Best based on ANOVA F-Value Score per feature                 
###############################################################################
plt.clf()
selector = SelectKBest(f_classif, k=len(pred_df.columns)) #####------SelectKBest features based on ANOVA F-Value Score (k=all)
selector.fit(pred_df,target_df)
scores = -np.log10(selector.pvalues_)
N=len(scores)
x=range(N)
median = statistics.median(scores)
#quartile1st = np.percentile(scores, 25, axis=0)
#quartile3rd = np.percentile(scores, 75)
#top17cutoff = np.percentile(scores, 40)
#plt.axvline(x=median)
#plt.axvline(x=top17cutoff)
#plt.axvline(x=quartile3rd)
plt.barh(x,scores)
plt.yticks(range(len(list(pred_df.columns))), list(pred_df.columns), rotation='horizontal')
plt.show()

    
#plt.clf()
#BOXPLOT Investigate prediction scores
scores.sort()
plt.boxplot(scores)
#plt.show()

###############################################################################
#              Correlation Matrix - NIF's                  
###############################################################################

import seaborn as sns
plt.clf()
sns.set(style="white")

# Compute the correlation matrix
corr = pred_df.astype(float).corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

#   Create legend
plot_legend = {}
for all in list(corr.columns):
    index = list(corr.columns).index(all)
    plot_legend[index] = all
       
yticks = xticks = list(corr.columns) #Use pre FS
#yticks_FS = xticks_FS = [0,1,2,3,4,5,6,7,8,11,12,15,17,19,20,25,26,28,29,30,31,32,33] #Use post FS

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, xticklabels=yticks,yticklabels=yticks,mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=1, cbar_kws={"shrink": .5})
g.set_yticklabels(g.get_yticklabels(),rotation = 0, fontsize = 10)
g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 10)
    

###############################################################################
#               Classifier build & analysis                 
###############################################################################
#Fresh results table
results_prf = []

#Classifers
dt_clf = DecisionTreeClassifier(random_state=0)
lr_clf = LogisticRegression()
gnb_clf = GaussianNB()
knn_clf = KNeighborsClassifier()
lda_clf = LinearDiscriminantAnalysis()

#Fn used to produce PRF scores and implement 10X CV
from sklearn.model_selection import KFold # import KFold
cv=10 #Input CV
kf = KFold(n_splits=cv)
def f1_score_fn(y,X,clf):
    warnings.filterwarnings("ignore")
    X = np.array(X)
    y = np.array(y)
    Pre_Rec_Fscore_avg = []
    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        Pre_Rec_Fscore = precision_recall_fscore_support(y_test, predictions, average='weighted')
        Pre_Rec_Fscore_avg.append(Pre_Rec_Fscore[:3])
        
    Pre_Avg_List = []
    Rec_Avg_List = []
    FScore_Avg_List = []
    for all in Pre_Rec_Fscore_avg:
            Pre_Avg_List.append(all[0])
            Rec_Avg_List.append(all[1])
            FScore_Avg_List.append(all[2])
 
    Pre_Avg = np.mean(Pre_Avg_List)
    Rec_Avg = np.mean(Rec_Avg_List)
    FScore_Avg = np.mean(FScore_Avg_List)
    return Pre_Avg,Rec_Avg,FScore_Avg

###############################################################################
#               DATASET 1    
#               With all features, all target values and all scripts
###############################################################################

results_prf = []
results_prf += [list(f1_score_fn(target_df,pred_df,dt_clf))]
results_prf += [list(f1_score_fn(target_df,pred_df,lr_clf))]
results_prf += [list(f1_score_fn(target_df,pred_df,gnb_clf))]
results_prf += [list(f1_score_fn(target_df,pred_df,knn_clf))]

d1targ = list(target_df)
groupByUpdateD1 = pd.DataFrame(d1targ)
gbD1 = groupByUpdateD1.groupby(groupByUpdateD1[0]).size()

###############################################################################
#               DATASET 2    
#               Ctrl Vs Prob ONLY
###############################################################################

newTarget = [] #Other,ProbAD,Ctrl

nonCtrlProbAD = [] #list of non-CtRL/PossAD =72 approx 13%
ctrlOnly = []
ProbADOnly = []
PossADOnly = []
AllAD_Ctrl = []

ctrlProbOnly = ctrlOnly.append(ProbADOnly)

i=0
while(i < len(target_df)):
    if target_df[i] == 'Control':
        newTarget.append((target_df.index[i], target_df[i]))
        ctrlOnly.append(target_df.index[i])
        AllAD_Ctrl.append((target_df.index[i], target_df[i]))
        i=i+1
    elif target_df[i] == 'ProbableAD':
        newTarget.append((target_df.index[i], target_df[i]))
        ProbADOnly.append(target_df.index[i])
        AllAD_Ctrl.append((target_df.index[i], 'AD')) 
        i=i+1
    elif target_df[i] == 'PossibleAD':
        newTarget.append((target_df.index[i], target_df[i]))
        PossADOnly.append(target_df.index[i]) 
        AllAD_Ctrl.append((target_df.index[i], 'AD')) 
        i=i+1
    else:
        newTarget.append((target_df.index[i],'Other'))
        nonCtrlProbAD.append(target_df.index[i])
        i=i+1
   
CtrlProbOnly_pre = totalScriptsDF.drop(nonCtrlProbAD)
held_out = CtrlProbOnly_pre.sample(frac=.1)
held_out_ind_list = list(held_out.index)
CtrlProbOnly_no_test = CtrlProbOnly_pre.drop(held_out_ind_list)

CtrlProbOnly = CtrlProbOnly_pre

CP_target_df = CtrlProbOnly['dx']
CP_pred_df = CtrlProbOnly[feature_names_pred]


results_prf += [list(f1_score_fn(CP_target_df,CP_pred_df,dt_clf))]
results_prf += [list(f1_score_fn(CP_target_df,CP_pred_df,lr_clf))]
results_prf += [list(f1_score_fn(CP_target_df,CP_pred_df,gnb_clf))]
results_prf += [list(f1_score_fn(CP_target_df,CP_pred_df,knn_clf))]

d2targ = list(CP_target_df)
groupByUpdateD2 = pd.DataFrame(d2targ)
gbD2 = groupByUpdateD1.groupby(groupByUpdateD2[0]).size()

###############################################################################
#               DATASET 3    
#               
###############################################################################
newTargDF = pd.DataFrame(newTarget)
newTargDF1 = newTargDF.rename(newTargDF[0])
newTargDF2 = newTargDF1.drop(0, axis=1)
newTargDF3 = newTargDF2[1]

cross_val_score(dt_clf, pred_df, newTargDF3, cv=10).mean()
cross_val_score(lr_clf, pred_df, newTargDF3, cv=10).mean()
cross_val_score(gnb_clf, pred_df, newTargDF3, cv=10).mean()
cross_val_score(knn_clf, pred_df, newTargDF3, cv=10).mean()

results_prf += [list(f1_score_fn(newTargDF3,pred_df,dt_clf))]
results_prf += [list(f1_score_fn(newTargDF3,pred_df,lr_clf))]
results_prf += [list(f1_score_fn(newTargDF3,pred_df,gnb_clf))]
results_prf += [list(f1_score_fn(newTargDF3,pred_df,knn_clf))]


d3targ = list(newTargDF3)
groupByUpdateD3 = pd.DataFrame(d3targ)
gbD3 = groupByUpdateD1.groupby(groupByUpdateD3[0]).size()

###############################################################################
#               DATASET 4  
#               Merge both types of AD's (Poss & Prob = 'All AD')
###############################################################################
newTargDF_AD = pd.DataFrame(AllAD_Ctrl)
newTargDF1_AD = newTargDF_AD.rename(newTargDF_AD[0])
newTargDF2_AD = newTargDF1_AD.drop(0, axis=1)
newTargDF3_AD = newTargDF2_AD[1]

cross_val_score(dt_clf, CP_pred_df, newTargDF3_AD, cv=10, scoring = 'f1_weighted').mean()
cross_val_score(lr_clf, CP_pred_df, newTargDF3_AD, cv=10, scoring = 'f1_weighted').mean()
cross_val_score(gnb_clf, CP_pred_df, newTargDF3_AD, cv=10, scoring = 'f1_weighted').mean() 
cross_val_score(knn_clf, CP_pred_df, newTargDF3_AD, cv=10, scoring = 'f1_weighted').mean()

results_prf += [list(f1_score_fn(newTargDF3_AD,CP_pred_df,dt_clf))]
results_prf += [list(f1_score_fn(newTargDF3_AD,CP_pred_df,lr_clf))]
results_prf += [list(f1_score_fn(newTargDF3_AD,CP_pred_df,gnb_clf))]
results_prf += [list(f1_score_fn(newTargDF3_AD,CP_pred_df,knn_clf))]

d4targ = list(newTargDF3_AD)
groupByUpdateD4 = pd.DataFrame(d4targ)
gbD4 = groupByUpdateD1.groupby(groupByUpdateD4[0]).size()