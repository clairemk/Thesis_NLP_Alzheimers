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
    
Encoding of Interactional Features.

Each transcript was inputted one at a time. 
A set of variables was created, one per transcipt, in the form of a Pandas series.
All series were merged at the end to created one DataFrame.
This DataFrame is then saved as a pickle, where it is then transposed and used for FS & Model building.

"""
###############################################################################
#               Libraries imported
###############################################################################
from nltk.tag import StanfordPOSTagger
from nltk.parse import stanford
from nltk.stem.wordnet import WordNetLemmatizer
import os
from nltk.parse.stanford import StanfordDependencyParser
from nltk.internals import find_jars_within_path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.feature_extraction import text
import nltk

###############################################################################
#               Used to Pre Process
###############################################################################

english_vocab = set(w.lower() for w in nltk.corpus.words.words())
lmtzr = WordNetLemmatizer()
my_stop_words = text.ENGLISH_STOP_WORDS

###############################################################################
#   Create Final Data Frame - will hold variables created per transcript
###############################################################################
df = pd.DataFrame([])

###############################################################################
#               Import Stanford Tagger & Parser 
###############################################################################

_stanford_url = 'https://nlp.stanford.edu/software/lex-parser.shtml'
jar = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-postagger-2015-12-09/stanford-postagger.jar'
model = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-postagger-2015-12-09/models/english-left3words-distsim.tagger'

pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')
stanford_dir = pos_tagger._stanford_jar.rpartition('/')[0]
stanford_jars = find_jars_within_path(stanford_dir)
pos_tagger._stanford_jar = ':'.join(stanford_jars)

_MAIN_CLASS = 'edu.stanford.nlp.parser.lexparser.    '
os.environ['STANFORD_PARSER'] = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'
os.putenv("CLASSPATH", "/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar")

path_to_jar_p = "/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser.jar"
path_to_models_jar_p = "/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar"
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar_p, path_to_models_jar=path_to_models_jar_p)
parser = stanford.StanfordParser(model_path="/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/lexparser.sh")

###############################################################################
#               In-Directory
###############################################################################

indir = '/Users/clairekelleher/Desktop/Thesis/Data/PItt_cookie_all_test' #<<<<< Set to directory where transcripts are <<<<<<<# 

###############################################################################
#               Main: Loops over each transcript once in indir 
###############################################################################
for root, dirs, filenames in os.walk(indir):
    try:
        for fname in filenames[1:]:
            fname = str(fname)
###############################################################################
#   Initial all variables/dictionaries/lists used through out script:
###############################################################################
            transcript = []
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
            
            postFillerList = []
            
            totalWordsPAR = 0
            
            turnCountPAR = 0
            turnCountINV = 0
            
            prolongationStutter = 0     #Place after prolonged segment eg. s:paghetti                
            brokenWordStutter = 0     #Pause within word eg. spa^ghetti
            blockCStutter = 0      #A block before word onset eg. ≠butter
            reptitionIterStutter = 0      #The curly left arrow brackets the repetition; iterations are marked with hyphens
            phonologicalFragmentStutter = 0      #Changes from “snake” to “dog” eg. &+sn dog
#            ###### ---- Typical Disfluencies ---- ######
            wholeWordRepetitionTypDys = 0 #Repeated word counts once eg. butter [/] butter
            multWholeWordRepetitionTypDys = 0 #butter [x 7] - Indicates that the word ‘butter’ was repeated seven times
            phraseRepTypDys = 0 # <that is a> [/] that is a dog.
            revisionCountsOnce = 0 #a dog [//] beast - Could count as Self Repair??****
            shortPause = 0
            medPause = 0
            longPause = 0
            pauseLength = 0
            allPauseCount = 0
            
            laughs = 0
            sighs = 0
            
            trailingOff = 0
            dunnoCount = 0
            
            backChannelmhm = 0 #from INV
            totalUtterances = 0 #from both INV and PAR
            sim_array_post_fil = 0
            
            ansWordLen = 0
            quesCountINV = 0
            quesCountPAR = 0
            avgAnsWordLen = 0
            parWordCount = 0
            
            allUtterances = []
###############################################################################
#   Open file path to begin loop:
###############################################################################
            with open(os.path.join(root, fname), 'r', encoding='utf8') as f:
                for line in f:
                    if line.startswith("*"): #Filter out any CHAT morphology
                        allUtterances.append(line)
            #************************************************
            #       Target variable (diagnosis) definitions 
            #************************************************
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

                    totalUtterances += 1
                    if line.startswith("*PAR:"): #For particpant only:
                        transcript += [line.split(" ")]
                        turnCountPAR += 1 #Particpant Turn Count
                        if "don't know" or "dunno" or "not sure" or "dont know" or "I can't" or "I'm not" in line:
                            dunnoCount += 1
                        words = line.split(" ")
                        
            #************************************************
            #       Count of CHAT Manual annotations used (For participant only)
            #************************************************
                        for index, word in enumerate(words):
                            for all in words:
                                if all.startswith("0"):
                                    omitted += 1
                                if "(" and ")" in all:
                                    incompleteWords += 1
                                if "xxx" in all:
                                    unintelligible += 1
                                if all.startswith("0"):
                                    omitted += 1                  
                                if all.startswith("+/?"): #Interruption of a Question
                                    Interruption_Q += 1
                                if all.startswith("+//."):
                                    Interruption_SI += 1      #in recall                   
    #            ###### ---- Stuttering-like Disfluencies ---- ######
                                if ":" in all:
                                    prolongationStutter += 1      #Place after prolonged segment eg. s:paghetti                
                                if "^" in all:
                                    brokenWordStutter += 1      #Pause within word eg. spa^ghetti
                                if "≠" in all:
                                    blockCStutter += 1      #A block before word onset eg. ≠butter
                                if "↫" and "- "in all:
                                    reptitionIterStutter += 1      #The curly left arrow brackets the repetition; iterations are marked with hyphens
                                if "&+"in all:
                                    phonologicalFragmentStutter += 1      #Changes from “snake” to “dog” eg. &+sn dog
    #            ###### ---- Typical Disfluencies ---- ######
                                if all == "[/]":
                                    wholeWordRepetitionTypDys += 1 #Repeated word counts once eg. butter [/] butter
                                if all.startswith("[x"):
                                    if len(all) > 2:
                                        print(all)
                                        ind_p1 = words[index+1]
                                        pre_result = re.findall(r'\d+',ind_p1)
                                        result = int(pre_result[0])
                                        multWholeWordRepetitionTypDys += result #butter [x 7] - Indicates that the word ‘butter’ was repeated seven times
                                    else:
                                        multWholeWordRepetitionTypDys += 1
                                if all.startswith("<"):
                                    phraseRepTypDys += 1 # <that is a> [/] that is a dog.
                                if all == "[//]":
                                    revisionCountsOnce += 1 #a dog [//] beast - Could count as Self Repair??****
                                if all == "(.)":
                                    shortPause += 1
                                    pauseLength += 1
                                if all == "(..)":
                                    medPause += 1
                                    pauseLength += 2
                                if all == "(...)":
                                    longPause += 1
                                    pauseLength += 3
                                if all == "(.)" or all == "(..)"== "(...)":
                                    allPauseCount += 1                                
                                if "&=laughs" in all:
                                    laughs += 1
                                if "&=sighs" in all:
                                    sighs += 1         
                                if "+..." in all:
                                    trailingOff += 1 
                                
                        senLen = len(words)
                        i=0
                        while(i < senLen):
                            if(words[i]=='' or words[i].isdigit() or words[i]=='exc'):
                                del words[i]
                                i=i
                                senLen = len(words)
                            else:
                                i=i+1
                                senLen = len(words)                        
                        
            #************************************************
            #Create a list that contains [pre_filler_term, filler_term, post_filler_term]
            #************************************************ 
        
            #************************************************
            #Filler Counter - PAR - Count words that begin with "&" - CHAT Manual 
            #   
            #    Disfluencies such as fillers, phonological fragments,
            #   and repeated segments are all coded by a preceding &.
            #************************************************
                        
                        for index, word in enumerate(words):
                            words[0] = words[0].replace("*PAR:\t","") 
                            if word.startswith("&") and "=" not in word:
                                if word.startswith("."):
                                    del word
                                if word == (" "):
                                    del word
                                word = re.sub(r'[^\w]', '', word)
                                fil_ind = index
                                
                                FillerLocoIndexSum += fil_ind
                                         
                                if fil_ind != 0: #If filler not at the start
                                    pre_fil_ind = fil_ind-1
                                else:
                                    pre_fil_ind = None
                                    
                                if fil_ind != len(words)-1: #If filler not at the end
                                    post_fil_ind = fil_ind+1
                                else:
                                    post_fil_ind = None                            
                                
                                if pre_fil_ind != None and post_fil_ind != None:
                                    fillerTrioList = [words[pre_fil_ind],words[fil_ind],words[post_fil_ind]]
                                    
                                if pre_fil_ind == None and post_fil_ind != None:
                                    fillerTrioList = [words[fil_ind],words[post_fil_ind]]
                                    
                                if post_fil_ind == None and pre_fil_ind != None:
                                    fillerTrioList = [words[pre_fil_ind],words[fil_ind]]                                    
                                    
                                pos_words = pos_tagger.tag(fillerTrioList)

                                if len(fillerTrioList) == 3:
                                    preFillerTags = pos_words[0][1]
                                    postFillerTags = pos_words[2][1]                                        
                                    postFillerList += [str(fillerTrioList[2])]
                                    
                                if len(fillerTrioList) == 2: 
                                    if pos_words[1][0] == words[fil_ind]:#if filler at the end then no post filler
                                        preFillerTags = pos_words[0][1]
                                    else:#if filler at the start 
                                        postFillerTags = pos_words[1][1]
                                        postFillerList += [str(fillerTrioList[1])]

            #************************************************
            #Find the POS tag post Filler Term
            #************************************************ 
                                if postFillerTags.startswith('J'):
                                    postFillerPOS_Adjective += 1
                                if postFillerTags.startswith('N'):
                                    postFillerPOS_Noun += 1                                                                
                                if postFillerTags.startswith('R'):
                                    postFillerPOS_Adverb += 1
                                if postFillerTags.startswith('V'):
                                    postFillerPOS_Verb += 1
                                else:
                                    postFillerPOS_Other += 1

                                words_only_FL = []
                                
            #************************************************
            #Avg Cosine Similarity of Word post Filler Term
            #************************************************      
                                i=0
                                while(i<len(postFillerList)):
                                    x = re.sub('[\s]', '_',postFillerList[i])
                                    words_only_FL += [re.sub('[^\w]', '', x)]
                                    i=i+1
                                 
                                if len(words_only_FL) > 1:
                                    for all in words_only_FL:
                                        if all in english_vocab and len(all)>1 and word not in my_stop_words:
                                            vect = TfidfVectorizer(stop_words=None)
                                            tfidf = vect.fit_transform(words_only_FL)
                                            sim_array = (tfidf * tfidf.T).A
                                            sim_array_post_fil = (sum(sum(sim_array))-sim_array.trace())/len(postFillerList)
                                        else:
                                            sim_array_post_fil += 0
                                else:
                                    sim_array_post_fil += 0
                                
                                
            #************************************************
            #   Create dictionary of pre filler term and post filler term
            #************************************************
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
            #************************************************
            #   Filler counts
            #************************************************       
                                fillerCountPAR += 1/senLen #Normalised by words in sentence
                                fillerCountPAR_UnNorm =+ 1
            #************************************************
            #  Count of each filler
            #************************************************     
                                if word not in fillerDict:
                                    fillerDict[word] = 1
                                else:
                                    fillerDict[word] += 1                                    
                                    
#                            Patters in responces - Did they answer the question? - Pull out key words......
            #************************************************
            #  Invigilator analysis
            #************************************************     
                    if line.startswith("*INV:"):
                        turnCountINV += 1

                        if "mhm" in line:
                            backChannelmhm += 1 #Backchannels used
                            
                    if line.startswith("*PAR"):
                        x = re.sub('[\s]', '_',line)
                        words_only = re.sub('[^\w]', '', x).split("_")                                    
                        totalWordsPAR += len(words_only)
                        
            #************************************************
            #   Location of filler in the sentence  
            #   Theory: Fillers will happen earlier in utterance for AD patients. 
            #   Sum of index will be low if fillers at start of sentence. 
                #Normalised by total number of fillers in transcripts 
            #************************************************  
                 
            DistinctFillerList = list(fillerDict.keys())

            if fillerCountPAR_UnNorm != 0:
                LocToFilRatio = FillerLocoIndexSum/fillerCountPAR_UnNorm 
            else:
                LocToFilRatio = 999
            
            #************************************************
            #       Avg Answer Length & Count of questions
            #************************************************ 
#            i=0
#            while(i < len(allUtterances)-1):
#                words = allUtterances[i].split(" ")
#                if(words[0].startswith("*INV") and "?" in words):
#                    quesCountINV += 1
#                    ansWordLen += len(allUtterances[i+1])
#                if(words[0].startswith("*PAR") and "?" in words):
#                    quesCountPAR += 1
#                elif(words[0].startswith("*PAR")):
#                    parWordCount += len(allUtterances)
#                i=i+1
#            
#            if quesCountINV != 0:
#                avgAnsWordLen = ansWordLen/quesCountINV
#            else:
#                avgAnsWordLen = parWordCount

            

            #************************************************
            #       Normalise each variable
            #************************************************ 
            
            incompleteWordsRatio = incompleteWords/totalWordsPAR
            turnCountRatio = turnCountINV/turnCountPAR #Higher => More like AD...
            prolongationStutter = prolongationStutter/totalWordsPAR    #Place after prolonged segment eg. s:paghetti                
            brokenWordStutter = brokenWordStutter/totalWordsPAR    #Pause within word eg. spa^ghetti
            blockCStutter = blockCStutter/totalWordsPAR     #A block before word onset eg. ≠butter
            reptitionIterStutter = reptitionIterStutter/totalWordsPAR     #The curly left arrow brackets the repetition; iterations are marked with hyphens
            phonologicalFragmentStutter = phonologicalFragmentStutter/totalWordsPAR     #Changes from “snake” to “dog” eg. &+sn dog
            wholeWordRepetitionTypDys = wholeWordRepetitionTypDys/totalWordsPAR#Repeated word counts once eg. butter [/] butter
            multWholeWordRepetitionTypDys = multWholeWordRepetitionTypDys/totalWordsPAR #butter [x 7] - Indicates that the word ‘butter’ was repeated seven times
            phraseRepTypDys = phraseRepTypDys/totalWordsPAR# <that is a> [/] that is a dog.
            revisionCountsOnce = revisionCountsOnce/totalWordsPAR#a dog [//] beast - Could count as Self Repair??****
            shortPause = shortPause/totalWordsPAR
            medPause = medPause/totalWordsPAR
            longPause = longPause/totalWordsPAR
            laughs = laughs/totalWordsPAR
            sighs = sighs/totalWordsPAR
            trailingOff = trailingOff/totalWordsPAR
            dunnoCount = dunnoCount/totalWordsPAR
            backChannelmhm = backChannelmhm/totalUtterances
            sim_array_post_fil = sim_array_post_fil*fillerCountPAR_UnNorm
            unintelligible = unintelligible/totalWordsPAR
            allPauseCount=allPauseCount/totalWordsPAR
            pauseLength = pauseLength/totalWordsPAR
            if fillerCountPAR_UnNorm != 0:
                postFillerPOS_Other = postFillerPOS_Other/fillerCountPAR_UnNorm
                postFillerPOS_Verb = postFillerPOS_Verb/fillerCountPAR_UnNorm
                postFillerPOS_Adjective = postFillerPOS_Adjective/fillerCountPAR_UnNorm
                postFillerPOS_Noun = postFillerPOS_Noun/fillerCountPAR_UnNorm
                postFillerPOS_Adverb = postFillerPOS_Adverb/fillerCountPAR_UnNorm
            if quesCountINV != 0:
                avgAnsWordLen = ansWordLen/quesCountINV
            else:
                avgAnsWordLen = avgAnsWordLen
                
            #************************************************
            #       Name variables
            #************************************************ 
            feats_name = [  'dx',
                            'fillerCountPAR',
                            'incompleteWordsRatio',
                            'unintelligible',
                            'omitted',
                            'Interruption_Q',
                            'postFillerPOS_Adjective',
                            'postFillerPOS_Noun',
                            'postFillerPOS_Adverb',
                            'postFillerPOS_Verb',
                            'postFillerPOS_Other',
                            'LocToFilRatio',
                            'turnCountRatio',
                            'prolongationStutter',
                            'brokenWordStutter',
                            'blockCStutter',
                            'reptitionIterStutter',
                            'phonologicalFragmentStutter',
                            'wholeWordRepetitionTypDys',
                            'multWholeWordRepetitionTypDys',
                            'phraseRepTypDys',
                            'revisionCountsOnce',
                            'shortPause',
                            'medPause',
                            'longPause',
                            'pauseLength',
                            'laughs',
                            'sighs',
                            'trailingOff',
                            'dunnoCount',
                            'backChannelmhm',
                            'sim_array_post_fil',
                            'allPauseCount',
                            'avgAnsWordLen2',
                            'quesCountINV2',
                            'quesCountPAR2',
                            'ansWordLen2',
                            'parWordCount']
            #************************************************
            #       Create series of variables
            #************************************************               
            series = pd.Series([dx,
                            fillerCountPAR,
                            incompleteWordsRatio,
                            unintelligible,
                            omitted,
                            Interruption_Q,
                            postFillerPOS_Adjective,
                            postFillerPOS_Noun,
                            postFillerPOS_Adverb,
                            postFillerPOS_Verb,
                            postFillerPOS_Other,
                            LocToFilRatio,
                            turnCountRatio,
                            prolongationStutter,
                            brokenWordStutter,
                            blockCStutter,
                            reptitionIterStutter,
                            phonologicalFragmentStutter,
                            wholeWordRepetitionTypDys,
                            multWholeWordRepetitionTypDys,
                            phraseRepTypDys,
                            revisionCountsOnce,
                            shortPause,
                            medPause,
                            longPause,
                            pauseLength,
                            laughs,
                            sighs,
                            trailingOff,
                            dunnoCount,
                            backChannelmhm,
                            sim_array_post_fil,
                            allPauseCount,
                            avgAnsWordLen,
                            quesCountINV,
                            quesCountPAR,
                            ansWordLen,
                            parWordCount], 
                                name = fname, index = feats_name)
            df = pd.concat([df, series], axis=1)
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

#df.to_pickle('/Users/clairekelleher/Desktop/Thesis/Data/new_IF_all.pkl')