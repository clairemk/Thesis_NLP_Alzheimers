#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:45:09 2017

@author: clairekelleher
"""

#----------------------------------------------------------------------------------------------------------------------------




# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#************************** --Libraries-- ******************************
from nltk.tag import StanfordPOSTagger
from nltk.parse import stanford
from nltk.stem.wordnet import WordNetLemmatizer
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import words as wd
from nltk.parse.stanford import StanfordDependencyParser
from nltk.internals import find_jars_within_path
from nltk.corpus import wordnet as wn
import pandas as pd

df = pd.DataFrame([])


_stanford_url = 'https://nlp.stanford.edu/software/lex-parser.shtml'
jar = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-postagger-2015-12-09/stanford-postagger.jar'
model = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-postagger-2015-12-09/models/english-left3words-distsim.tagger'

pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')
stanford_dir = pos_tagger._stanford_jar.rpartition('/')[0]
stanford_jars = find_jars_within_path(stanford_dir)
pos_tagger._stanford_jar = ':'.join(stanford_jars)

_MAIN_CLASS = 'edu.stanford.nlp.parser.lexparser.LexicalizedParser'
os.environ['STANFORD_PARSER'] = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'
os.putenv("CLASSPATH", "/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar")

path_to_jar_p = "/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser.jar"
path_to_models_jar_p = "/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar"
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar_p, path_to_models_jar=path_to_models_jar_p)

#************************** --Parser-- ******************************
#To do: Sort out using STANFORD PARSER!

from nltk.chunk.regexp import RegexpParser

grammar = '''
    NP: {<DT>? <JJ>* <NN>*} # NP
    P: {<IN>}           # Preposition
    V: {<V.*>}          # Verb
    PP: {<P> <NP>}      # PP -> P NP
    VP: {<V> <NP|PP>*}  # VP -> V (NP|PP)*
'''
    
reg_parser = RegexpParser(grammar)
parser = stanford.StanfordParser(model_path="/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/lexparser.sh")
lmtzr = WordNetLemmatizer()

#----------------------------------------------------------------------------------------------------------------------------






import os
import pandas as pd

del df
df = pd.DataFrame([])

indir = '/Users/clairekelleher/Desktop/Thesis/Data/noControlTestFlu'

for root, dirs, filenames in os.walk(indir):
    try:
        for fname in filenames[1:]:  
            fname = str(fname)
            
            fillerCountPAR = 0 #Normalised per utterance
            fillerCountPAR2 = 0
            fillerCountPAR_UnNorm = 0 #Raw count of fillers
            
            incompleteWords = 0 #Includes shortenings of words. Eg. Travelling == Travellin'
            unintelligible = 0
            omitted = 0
            Interruption_Q = 0
            Interruption_SI = 0

            preFillPOS_Ct = {}
            postFillPOS_Ct = {}
            
            prePostFill_Tups = {}
            
            fillerDict = {} 
            
            postFillerPOS_Adjective = 0 #J
            postFillerPOS_Noun = 0 #N
            postFillerPOS_Adverb = 0 #RB
            postFillerPOS_Verb = 0 #V
            postFillerPOS_Other = 0
            
            FillerLocoIndexSum = 0
            
            totalWordsPAR = 0
            
            turnCountPAR = 0
            turnCountINV = 0
            
            with open(os.path.join(root, fname), 'r', encoding='utf8') as f:
                for line in f:
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
                            
            #************************************************
            #Filler Counter - PAR - Count words that begin with "&" - CHAT Manual 
            #************************************************
                    if line.startswith("*PAR:"):                        
                        words = line.split(" ")
                        senLen = len(words)
                        turnCountPAR += 1
#                        pos_words = pos_tagger.tag(words)
                        
                        for index, word in enumerate(words):
                            words[0] = words[0].replace("*PAR:\t","")
                            if word.startswith("&") and "=" not in word:
                                
                                fil_ind = index
                                FillerLocoIndexSum += fil_ind
                                         
                                if fil_ind != 0:
                                    pre_fil_ind = index-1
                                else:
                                    pre_fil_ind = None
                                    
                                if fil_ind != senLen-1:
                                    post_fil_ind = index+1
                                else:
                                    post_fil_ind = None                            
                                
                                if pre_fil_ind != None and post_fil_ind != None:
                                    fillerTrioList = [words[pre_fil_ind],words[fil_ind],words[post_fil_ind]]
                                    
                                if pre_fil_ind == None and post_fil_ind != None:
                                    fillerTrioList = [" ",words[fil_ind],words[post_fil_ind]]
                                    
                                if post_fil_ind == None and pre_fil_ind != None:
                                    fillerTrioList = [words[pre_fil_ind],words[fil_ind]," "]                                    
                                    
                                pos_words = pos_tagger.tag(fillerTrioList)
                                
                                preFillerTags = pos_words[0][1]
                                postFillerTags = pos_words[2][1]
                                pairPrePost = {preFillerTags,postFillerTags}
                                prePostFill_List = list(pairPrePost)
                                
                                pairStr = str(pairPrePost)
                                
                                if pairStr in prePostFill_Tups:
                                    prePostFill_Tups[pairStr] += 1
                                else:
                                    prePostFill_Tups[pairStr] = 1
                                    
                                if len(prePostFill_List) == 2:
                                    if prePostFill_List[0] in preFillPOS_Ct:
                                        preFillPOS_Ct[prePostFill_List[0]] += 1
                                    else:
                                        preFillPOS_Ct[prePostFill_List[0]] = 1
                                    
                                    if prePostFill_List[1] in postFillPOS_Ct:
                                        postFillPOS_Ct[prePostFill_List[1]] += 1
                                    else:
                                        postFillPOS_Ct[prePostFill_List[1]] = 1
                                 
                                if len(prePostFill_List) == 1 and len(prePostFill_Tups) == 1:
#                                    print(prePostFill_List[0]) #Start of sentence - Eg. If the first word is "Uh" => Tag must be post filler
                                    if prePostFill_List[0] in postFillPOS_Ct:
                                        postFillPOS_Ct[prePostFill_List[0]] += 1
                                    else:
                                        postFillPOS_Ct[prePostFill_List[0]] = 1
                                if len(prePostFill_List) == 1 and len(prePostFill_Tups) > 1:
                                    if prePostFill_List[0] in preFillPOS_Ct:
                                        preFillPOS_Ct[prePostFill_List[0]] += 1
                                    else:
                                        preFillPOS_Ct[prePostFill_List[0]] = 1
                                        
#                                if preFillerTags in preFillPOS_Ct:
#                                    preFillPOS_Ct[preFillerTags] += 1
#                                else:
#                                    preFillPOS_Ct[preFillerTags] = 1
#                                    
#                                if postFillerTags in postFillPOS_Ct:
#                                    postFillPOS_Ct[postFillerTags] += 1
#                                else:
#                                    postFillPOS_Ct[postFillerTags] = 1
                                                                
                                fillerCountPAR += 1/senLen
                                fillerCountPAR_UnNorm =+ 1
                                if word not in fillerDict:
                                    fillerDict[word] = 1
                                else:
                                    fillerDict[word] += 1
                                
                                
                            if word.startswith("&-") and "=" not in word:
                                fillerCountPAR2 += 1/senLen
                                if word not in fillerDict:
                                    fillerDict[word] = 1
                                else:
                                    fillerDict[word] += 1
                                    
                            totalWordsPAR += 1
#                            Patters in responces - Did they answer the question? - Pull out key words......
#                                   


                            
                            if "(" and ")" in word:
                                incompleteWords += 1
                            if "xxx" in word:
                                unintelligible += 1
                            if word.startswith("0"):
                                omitted += 1/senLen                  
                            if word.startswith("+/?"):
                                Interruption_Q += 1
                            if word.startswith("+//."):
                                Interruption_SI += 1      #in recall                   

                    if line.startswith("*INV:"):
                        turnCountINV += 1         
            DistinctFillerList = list(fillerDict.keys())
            
#           Return which POS tags appear the most AFTER a filler word 
#           "Most":         >=2 STD's over the distribution of POS tag frequencies - upon brief observation, 1 STD would include nearly everything 
        
            tagDistPostFil = pd.Series(list(postFillPOS_Ct.values()),index=[list(postFillPOS_Ct.keys())])       
            maxPOSfreg = tagDistPostFil.max()
   
            if(len(tagDistPostFil[tagDistPostFil == maxPOSfreg]) != len(tagDistPostFil)):    
                maxPostPOS = tagDistPostFil[tagDistPostFil == maxPOSfreg].index[0:len(tagDistPostFil)]
                i=0
                while(i<len(maxPostPOS)):              
                    HL_POS_max = maxPostPOS[i][0]
                    
                    if HL_POS_max == 'J':
                        postFillerPOS_Adjective += 1

                    if HL_POS_max == 'N':
                        postFillerPOS_Noun += 1                        
                        
                    if HL_POS_max == 'R' and maxPostPOS[i][0] == 'B':
                        postFillerPOS_Adverb += 1
                    
                    if HL_POS_max == 'V':
                        postFillerPOS_Verb += 1     
                    else:
                        postFillerPOS_Other += 1
                        
                    i=i+1
#            else:
#                print("Post Filler POS tag frequency of equal value") 
            
            if fillerCountPAR_UnNorm != 0:
                LocToFilRatio = FillerLocoIndexSum/fillerCountPAR_UnNorm #Theory: Fillers will happen earlier in utterance for AD patients. Sum of index will be low if fillers at start of sentence. Normalised by total number of fillers in transcripts 
#            => LOW LocToFilRatio = fillers at start of utts and alot of fillers in overall transcript => AD
            else:
                LocToFilRatio = 999
            
            incompleteWordsRatio = incompleteWords/totalWordsPAR
            turnCountRatio = turnCountINV/turnCountPAR #Higher => More like AD...
#            std = tagDistPostFil.std() 
#            if std != 0: #Ie. Not all the same value for POS count
            
            feats_name = ['dx', 
                          'fillerCountPAR', 
                          'fillerCountPAR2', 
                          'incompleteWordsRatio', 
                          'unintelligible', 
                          'omitted', 
                          'Interruption_Q', 
                          'Interruption_SI', 
                          'postFillerPOS_Adjective', 
                          'postFillerPOS_Noun', 
                          'postFillerPOS_Adverb', 
                          'postFillerPOS_Verb', 
                          'postFillerPOS_Other', 
                          'LocToFilRatio',
                          'turnCountPAR',
                          'turnCountINV',
                          'turnCountRatio']
            series = pd.Series([dx, 
                                fillerCountPAR, 
                                fillerCountPAR2, 
                                incompleteWordsRatio, 
                                unintelligible, 
                                omitted, 
                                Interruption_Q, 
                                Interruption_SI,
                                postFillerPOS_Adjective,
                                postFillerPOS_Noun,
                                postFillerPOS_Adverb,
                                postFillerPOS_Verb, 
                                postFillerPOS_Other, 
                                LocToFilRatio,
                                turnCountPAR,
                                turnCountINV,
                                turnCountRatio], 
                                name = fname, index = feats_name)
            df = pd.concat([df, series], axis=1)
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

print(df)
                        
#testlist = words
#for position, item in enumerate(testlist):
#    print(position)
# Look at similarity of words after filler - see if they pause/forget similar words.