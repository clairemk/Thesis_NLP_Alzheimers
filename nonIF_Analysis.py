#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:26:40 2017

@author: clairekelleher
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV #http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D


df1 = pd.read_pickle('/Users/clairekelleher/Desktop/Thesis/Data/df0_to_2451.pkl')
df2 = pd.read_pickle('/Users/clairekelleher/Desktop/Thesis/Data/df2470_to_eof.pkl')

df1t = df1.transpose()
df2t = df2.transpose()

frames = [df1t,df2t]

feature_names_all = ["dx","Adverb_Ct_Total","Avg_Word_Len","INV_Word_Ct","INVtoPAR_word_ratio","Keyword_cookie","Keyword_counter","Keyword_curtain","Keyword_sink","Keyword_stool","Keyword_window","Noun_Ct_Total","PAR_UTT_ct","PAR_Word_Ct","ProNoun_Ct","ProNoun_Ct","ProNoun_Noun_Ratio","Script_Letter_Ct","Script_Word_Ct","Subj_IU_Total","Total_Keywords_Mentioned","Verb_Ct_3sing","Verb_Ct_Base","Verb_Ct_Gerund","Verb_Ct_Past","Verb_Ct_Past_P","Verb_Ct_Past_T","Verb_Ct_Total","Verb_Ct_non3sing","cos_avg","wordsOnceOverTotal","wordsUsedOnce"] #Rename feature names

feature_names_all_1ST_RUN = ["dx","Adverb_Ct_Total","Avg_Word_Len","INV_Word_Ct","Keyword_cookie","Keyword_counter","Keyword_curtain","Keyword_sink","Keyword_stool","Keyword_window","Noun_Ct_Total","PAR_UTT_ct","PAR_Word_Ct","ProNoun_Ct","ProNoun_Noun_Ratio","Script_Letter_Ct","Script_Word_Ct","Subj_IU_Total","Total_Keywords_Mentioned","Verb_Ct_3sing","Verb_Ct_Base","Verb_Ct_Gerund","Verb_Ct_Past","Verb_Ct_Past_P","Verb_Ct_Past_T","Verb_Ct_Total","Verb_Ct_non3sing","cos_avg","wordsOnceOverTotal","wordsUsedOnce"] #Rename feature names
#^^^Count: 29^^^#
feature_names_all_POST_FEAT_RED = ["dx","Adverb_Ct_Total","Avg_Word_Len","INV_Word_Ct","Keyword_cookie","Keyword_counter","Keyword_curtain","Keyword_sink","Keyword_stool","Keyword_window","Noun_Ct_Total","PAR_UTT_ct","PAR_Word_Ct","ProNoun_Ct","ProNoun_Noun_Ratio","Script_Letter_Ct","Script_Word_Ct","Subj_IU_Total","Total_Keywords_Mentioned","Verb_Ct_3sing","Verb_Ct_Base","Verb_Ct_Gerund","Verb_Ct_Past","Verb_Ct_Past_P","Verb_Ct_non3sing","cos_avg","wordsOnceOverTotal","wordsUsedOnce"] #Rename feature names
#^^^Count: 28^^^#
feature_names_all_POST_FEAT_RED_LessVerb = ["dx","Adverb_Ct_Total","Avg_Word_Len","INV_Word_Ct","Keyword_cookie","Keyword_counter","Keyword_curtain","Keyword_sink","Keyword_stool","Keyword_window","Noun_Ct_Total","PAR_UTT_ct","PAR_Word_Ct","ProNoun_Ct","ProNoun_Noun_Ratio","Script_Letter_Ct","Script_Word_Ct","Subj_IU_Total","Total_Keywords_Mentioned","Verb_Ct_Total","cos_avg","wordsOnceOverTotal","wordsUsedOnce"] #Rename feature names
#^^^Count: 23^^^#

feature_names_all_POST_FEAT_RED_LessVerb_addtt = ["dx","INVtoPAR_word_ratio","Adverb_Ct_Total","Avg_Word_Len","INV_Word_Ct","Keyword_cookie","Keyword_counter","Keyword_curtain","Keyword_sink","Keyword_stool","Keyword_window","Noun_Ct_Total","PAR_UTT_ct","PAR_Word_Ct","ProNoun_Ct","ProNoun_Noun_Ratio","Script_Letter_Ct","Script_Word_Ct","Subj_IU_Total","Total_Keywords_Mentioned","Verb_Ct_Total","cos_avg","wordsOnceOverTotal","wordsUsedOnce"] #Rename feature names


x_featList_to_use_x = feature_names_all_POST_FEAT_RED_LessVerb_addtt #######-------Insert Feat List from above here------######

feat_len = len(x_featList_to_use_x)
feature_names_pred = x_featList_to_use_x[1:feat_len]
#feature_names_pred_act = ["Adverb_Ct_Total","Avg_Word_Len","INV_Word_Ct","INVtoPAR_word_ratio","Keyword_cookie","Keyword_counter","Keyword_curtain","Keyword_sink","Keyword_stool","Keyword_window","Noun_Ct_Total","PAR_UTT_ct","PAR_Word_Ct","ProNoun_Ct","ProNoun_Ct","ProNoun_Noun_Ratio","Script_Letter_Ct","Script_Word_Ct","Subj_IU_Total","Total_Keywords_Mentioned","Verb_Ct_3sing","Verb_Ct_Base","Verb_Ct_Gerund","Verb_Ct_Past","Verb_Ct_Past_P","Verb_Ct_Past_T","Verb_Ct_Total","Verb_Ct_non3sing","cos_avg","wordsOnceOverTotal","wordsUsedOnce"] #Rename feature names
target = feature_names_all[0]

totalScriptsDF = pd.concat(frames)
totalScriptsDF.columns = feature_names_all
totalScriptsDF = totalScriptsDF.loc[:,~totalScriptsDF.columns.duplicated()]

target_df = totalScriptsDF[target]
pred_df = totalScriptsDF[feature_names_pred]

held_out = totalScriptsDF.sample(frac=.1)
xx = held_out.index
new_test = pred_df.drop(xx)

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),scoring='accuracy')

#optNumFeats = rfecv.fit(pred_df, target_df)


#Summary of targets:
totalScriptsDF.groupby(totalScriptsDF['dx']).size()

plt.clf()
# Perform feature selection
selector = SelectKBest(f_classif, k=feat_len-1)
selector.fit(pred_df,target_df)
#selector.fit(CP_pred_df,CP_target_df)

scores = -np.log10(selector.pvalues_)
N=len(scores)
x=range(N)
plt.bar(x,scores)
plt.xticks(range(len(feature_names_pred)), feature_names_pred, rotation='vertical')
plt.show()


#Features to do with describing play a signicant role in predicting = Number of adverbs, Keywords Mentioned, ProNoun_Ct
#INVtoPAR_word_ratio is the highest predictor for the target - INTERACTIONAL FEATURE :)

plt.clf()
#Investigate prediction scores - BOXPLOT
scores.sort()
plt.boxplot(scores)
plt.show()

mean = scores.mean()
kBest = sum(scores>=mean)
print('Number of selected features above mean:' + str(kBest))
highScore = []
for score in scores:
    if(score > mean):
        highScore.append(score)
        
len(highScore)

#########With all features, all target values and all scripts########################################
dt_clf = DecisionTreeClassifier(random_state=0)
lr_clf = LogisticRegression()
gnb_clf = GaussianNB()
knn_clf = KNeighborsClassifier()
lda_clf = LinearDiscriminantAnalysis()

cross_val_score(dt_clf, pred_df, target_df, cv=3).mean()
cross_val_score(lr_clf, pred_df, target_df, cv=3).mean()
cross_val_score(gnb_clf, pred_df, target_df, cv=3).mean()
cross_val_score(knn_clf, pred_df, target_df, cv=3).mean()
#cross_val_score(lda_clf, pred_df, target_df, cv=3).mean()

#------Results before reducing variables -- Pretending TT was never there (29)
#dt_clf = 0.48011060961880636
#lr_clf = 0.60188890024955599 --NOT BAD!
#gnb_clf = 0.28317295858279462
#knn_clf = 0.4381665398058841

#------Results after reducing variables  (28)
#dt_clf = 0.46715192944701145
#lr_clf = 0.60551219567613013
#gnb_clf = 0.28131208131208135
#knn_clf = 0.43633503797438222

#------Results after reducing verb break down variables (23)
#dt_clf = 0.46731076239272956
#lr_clf = 0.62906410283459469
#gnb_clf = 0.26687797507469641
#knn_clf = 0.43996834160768589

#----Try Above 23 Festures plus TT--- (24)
#dt_clf = 0.48539786408638869
#lr_clf = 0.62908411924805374
#gnb_clf = 0.27958937139265011
#knn_clf = 0.43996834160768589

################################################################################
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
        newTarget.append((target_df.index[i], target_df[i])) #totalScriptsDF_samp.index[i]
        ctrlOnly.append(target_df.index[i]) #totalScriptsDF_samp.index[i]
        AllAD_Ctrl.append((target_df.index[i], target_df[i])) #totalScriptsDF_samp.index[i]
        i=i+1
    elif target_df[i] == 'ProbableAD':
        newTarget.append((target_df.index[i], target_df[i])) #totalScriptsDF_samp.index[i]
        ProbADOnly.append(target_df.index[i]) #totalScriptsDF_samp.index[i]
        AllAD_Ctrl.append((target_df.index[i], 'AD')) #totalScriptsDF_samp.index[i]
        i=i+1
    elif target_df[i] == 'PossibleAD':
        newTarget.append((target_df.index[i], target_df[i])) #totalScriptsDF_samp.index[i]
        PossADOnly.append(target_df.index[i]) #totalScriptsDF_samp.index[i]
        AllAD_Ctrl.append((target_df.index[i], 'AD')) #totalScriptsDF_samp.index[i]
        i=i+1
    else:
        newTarget.append((target_df.index[i],'Other'))
        nonCtrlProbAD.append(target_df.index[i])
        i=i+1

#Count checks
groupByUpdate = pd.DataFrame(newTarget)
groupByUpdate1 = pd.DataFrame(AllAD_Ctrl)

gbu = groupByUpdate.groupby(groupByUpdate[1]).size()
aad_gbu = groupByUpdate1.groupby(groupByUpdate1[1]).size()


################################################################################      
CtrlProbOnly_pre = totalScriptsDF.drop(nonCtrlProbAD)
held_out = CtrlProbOnly_pre.sample(frac=.1)
held_out_ind_list = list(held_out.index)
CtrlProbOnly_no_test = CtrlProbOnly_pre.drop(held_out_ind_list)

CtrlProbOnly = CtrlProbOnly_pre

CP_target_df = CtrlProbOnly[target]
CP_pred_df = CtrlProbOnly[feature_names_pred]

cross_val_score(dt_clf, CP_pred_df, CP_target_df, cv=10).mean()
cross_val_score(lr_clf, CP_pred_df, CP_target_df, cv=10).mean()
cross_val_score(gnb_clf, CP_pred_df, CP_target_df, cv=10).mean()
cross_val_score(knn_clf, CP_pred_df, CP_target_df, cv=10).mean() 
#cross_val_score(lda_clf, CP_pred_df, CP_target_df, cv=10).mean()

#------Results before reducing variables -- Pretending TT was never there (29)
#dt_clf = 0.57080847723704875
#lr_clf = 0.68519466248037664 --LOOKING GOOD
#gnb_clf = 0.36399999999999999
#knn_clf = 0.47446781789638931

#------Results after reducing variables (28)
#dt_clf = 0.57499843014128726
#lr_clf = 0.68715384615384623 --SLIGHT IMPROVEMENT
#gnb_clf = 0.36600470957613812
#knn_clf = 0.47835007849293565

#------Results after reducing verb break down variables (23)
#dt_clf = 0.55671271585557303
#lr_clf = 0.7072354788069074 ----Woop!
#gnb_clf = 0.32736106750392457
#knn_clf = 0.47846781789638937

#----Try Above 23 Festures plus TT--- (24)
#dt_clf = 0.54456357927786503
#lr_clf = 0.69719466248037665
#gnb_clf = 0.34932496075353214
#knn_clf = 0.47846781789638937

###############################################################################
newTargDF = pd.DataFrame(newTarget)
newTargDF1 = newTargDF.rename(newTargDF[0])
newTargDF2 = newTargDF1.drop(0, axis=1)
newTargDF3 = newTargDF2[1]

cross_val_score(dt_clf, pred_df, newTargDF3, cv=10).mean()
cross_val_score(lr_clf, pred_df, newTargDF3, cv=10).mean()
cross_val_score(gnb_clf, pred_df, newTargDF3, cv=10).mean()
cross_val_score(knn_clf, pred_df, newTargDF3, cv=10).mean()
#cross_val_score(lda_clf, pred_df, newTargDF, cv=10).mean() #0.60544992531834629

#------Results before reducing variables -- Pretending TT was never there (29)
#dt_clf = 0.45585858585858585
#lr_clf = 0.61640543364681299
#gnb_clf = 0.28265180541042606
#knn_clf = 0.4358992221061187


#------Results after reducing variables (28)
#dt_clf = 0.46148148148148149
#lr_clf = 0.62189364913502843
#gnb_clf = 0.2827458492975734
#knn_clf = 0.43940787182166491

#------Results after reducing verb break down variables (23)
#dt_clf = 0.48148148148148151
#lr_clf = 0.64544293509810757 ---Improvement!
#gnb_clf = 0.25125159642401018
#knn_clf = 0.43771740392430053

#----Try Above 23 Festures plus TT--- (24)
#dt_clf = 0.43795193312434683
#lr_clf = 0.64538256124463023
#gnb_clf = 0.2767467781260885
#knn_clf = 0.43771740392430053

###############################################################################
held_out = CtrlProbOnly.sample(frac=.1)
held_out_ind_list = list(held_out.index)

#Ctrl_only_DF_hhtest = CtrlProbOnly_no_test.drop(held_out_ind_list).drop(ProbADOnly)
#Pron_only_DF_hhtest = CtrlProbOnly_no_test.drop(ctrlOnly)


################################################################################

#dforg = pd.DataFrame(totalScriptsDF)
#dforg.columns = len(feature_names_pred_act)
##CrtlOnlyDF = dforg.query('= Control')

################################################################################
#           Worried about co-linearity checking for co-linearirty
################################################################################

#Remove Verb_Ct_Past_T because it correlates strongly with Verb_Ct_Past
#Revoe Verb_Ct_Total or else break down of differnet types of verbs 
#Investigate Keyword count break down to KW ct
df = pred_df
foo = df.astype(float)
corr = foo.corr()
plt.matshow(corr)

######Feature removal: See feature_names_all_new

############   TRY MERGE BOTH TYPES OF AD'S #########################################

#List of index's = AllAD

newTargDF_AD = pd.DataFrame(AllAD_Ctrl)
newTargDF1_AD = newTargDF_AD.rename(newTargDF_AD[0])
newTargDF2_AD = newTargDF1_AD.drop(0, axis=1)
newTargDF3_AD = newTargDF2_AD[1]

cross_val_score(dt_clf, CP_pred_df, newTargDF3_AD, cv=10).mean()
cross_val_score(lr_clf, CP_pred_df, newTargDF3_AD, cv=10).mean()
cross_val_score(gnb_clf, CP_pred_df, newTargDF3_AD, cv=10).mean() 
cross_val_score(knn_clf, CP_pred_df, newTargDF3_AD, cv=10).mean()
#cross_val_score(lda_clf, CP_pred_df, newTargDF3_AD, cv=10).mean()

#------Results before reducing variables -- Pretending TT was never there (29)
#dt_clf = 0.64323649459783916 ---WOOP
#lr_clf = 0.72745018007202888 ---V Good***
#gnb_clf = 0.65928051220488193 ---FIRST DECENT RESULT FOR GNB! ---INTERESTING***
#knn_clf = 0.53070268107242891 

#------Results after reducing variables (28)
#dt_clf = 0.61931332533013195
#lr_clf = 0.72540936374549825 ---V Good***
#gnb_clf = 0.67336214485794321 ---IMPROVEMENT! 
#knn_clf = 0.53070268107242891

#------Results after reducing verb break down variables (23)
#dt_clf = 0.65907643057222876
#lr_clf = 0.7333669467787115 ---Highest yet!***
#gnb_clf = 0.69136534613845535 ---Best ever for GNB
#knn_clf = 0.53070268107242891

#----Try Above 23 Festures plus TT--- (24)
#dt_clf = 0.61124129651860737
#lr_clf = 0.72324769907963182
#gnb_clf = 0.70348779511804715 ---only improvement (slightly)
#knn_clf = 0.53070268107242891


held_out = CtrlProbOnly.sample(frac=.1)
held_out_ind_list = list(held_out.index)


#********* Saw some improvement in certain datasets accuracies but unsure if optimal variables were removed so used Feature ranking with recursive feature elimination.
#********* sklearn.feature_selection.RFE




