#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:45:09 2017

@author: Claire Mary Kelleher
MSc. Big Data Science 
Thesis Project: Interactional and Linguistic Analysis for Computationally Diagnosing Alzheimer’s Disease
"""

"""
Purpose of this script:
    
Encoding of Ans-Based Interactional Features. (Added to rest of IF's when converted from pickle for FS & Model building)

Each transcript was inputted one at a time. 
A set of variables was created, one per transcipt, in the form of a Pandas series.
All series were merged at the end to created one DataFrame.
This DataFrame is then saved as a pickle, where it is then transposed and used for FS & Model building.

"""


###############################################################################
#               Libraries imported
###############################################################################

from nltk.stem.wordnet import WordNetLemmatizer
import os
import pandas as pd
from sklearn.feature_extraction import text
import nltk

###############################################################################
#               Used to Pre Process
###############################################################################
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
lmtzr = WordNetLemmatizer()
my_stop_words = text.ENGLISH_STOP_WORDS


###############################################################################
#               Create Final Data Frame
###############################################################################
df = pd.DataFrame([])

###############################################################################
#               Main: Loop over each transcript once in indir 
###############################################################################

indir = '/Users/clairekelleher/Desktop/Thesis/Data/PItt_cookie_all_indf_IF' #<<<<< Set to directory where transcripts are <<<<<<<# 
transcriptCount = 0

for root, dirs, filenames in os.walk(indir):
    try:
        for fname in filenames[1:]:
            fname = str(fname)
            transcriptCount = transcriptCount+1

            quesCountINV = 0
            quesCountPAR = 0
            ansWordLen = 0
            avgAnsWordLen = 0
            parWordCount = 0
            allUtterances = []
            with open(os.path.join(root, fname), 'r', encoding='utf8') as f:
                for line in f:
                    if line.startswith("*"):
                        allUtterances.append(line)
                    if line.startswith("@ID:") and "PAR" in line:
                        info = line.split("|")
                        if 'ProbableAD' in info:
                            dx = 'ProbableAD'
                        if 'PossibleAD' in info:
                            dx = 'PossibleAD'
                        if 'MCI' in info:
                            dx = 'MCI'
                        if 'Memory' in info:
                            dx = 'Memory'
                        if 'Vascular' in info:
                            dx = 'Vascular'
                        if 'Control' in info:
                            dx = 'Control'
            
            i=0
            while(i < len(allUtterances)-1):
                words = allUtterances[i].split(" ")
                if(words[0].startswith("*INV") and "?" in words):
                    quesCountINV += 1
                    ansWordLen += len(allUtterances[i+1])
                if(words[0].startswith("*PAR") and "?" in words):
                    quesCountPAR += 1
                elif(words[0].startswith("*PAR")):
                    parWordCount += len(allUtterances)
                i=i+1
            
            if quesCountINV != 0:
                avgAnsWordLen = ansWordLen/quesCountINV
            else:
                avgAnsWordLen = parWordCount
            
            feats_name = ['avgAnsWordLen2',
                            'quesCountINV2',
                            'quesCountPAR2',
                            'ansWordLen2',
                            'parWordCount']
                          
            series = pd.Series([avgAnsWordLen,
                                quesCountINV,
                                quesCountPAR,
                                ansWordLen,
                                parWordCount], 
                                name = fname, index = feats_name)
            df = pd.concat([df, series], axis=1)
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

df.to_pickle('/Users/clairekelleher/Desktop/Thesis/Data/script_ansLen.pkl') #<<<<<<Converts data to a Python Pickle to be accesed during FS and model build.