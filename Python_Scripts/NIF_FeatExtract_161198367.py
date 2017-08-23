#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14 July 2017 at 12:29

@author: Claire Mary Kelleher
MSc. Big Data Science 
Thesis Project: Interactional and Linguistic Analysis for Computationally Diagnosing Alzheimer’s Disease
"""

"""
Purpose of this script:
    
Encoding of Non-Interactional Features.

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
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import words as wd
from nltk.parse.stanford import StanfordDependencyParser
from nltk.internals import find_jars_within_path
from nltk.corpus import wordnet as wn
import pandas as pd

df = pd.DataFrame([])

###############################################################################
#               Import Stanford Tagger
###############################################################################
_stanford_url = 'https://nlp.stanford.edu/software/lex-parser.shtml'
jar = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-postagger-2015-12-09/stanford-postagger.jar'
model = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-postagger-2015-12-09/models/english-left3words-distsim.tagger'

pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')
stanford_dir = pos_tagger._stanford_jar.rpartition('/')[0]
stanford_jars = find_jars_within_path(stanford_dir)
pos_tagger._stanford_jar = ':'.join(stanford_jars)

###############################################################################
#               Import Stanford Parser 
###############################################################################
_MAIN_CLASS = 'edu.stanford.nlp.parser.lexparser.LexicalizedParser'
os.environ['STANFORD_PARSER'] = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'
os.putenv("CLASSPATH", "/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar")

path_to_jar_p = "/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser.jar"
path_to_models_jar_p = "/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar"
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar_p, path_to_models_jar=path_to_models_jar_p)


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

#def file_len(fname):
#    with open(fname) as f:
#        for i, l in enumerate(f):
#            pass
#        return i + 1
#********************** --Create read in fn-- ******************

#fname = "002-0.cex"
indir = '/Users/clairekelleher/Desktop/Thesis/Data/PItt_cookie_all_test'

for root, dirs, filenames in os.walk(indir):
    try:
        for fname in filenames[1:]:  
            fname = str(fname)
            
            
    #        line_ct = fxile_len(fname) #Total lines in file
            
            
            
            #********************--Dictionarys etc--********************      
            distinctWordCt = {} #Used for HS
            POS_COUNT = {}
            
            Script_Word_Ct = 0
            Script_Letter_Ct = 0
            INV_Word_Ct = 0
            PAR_Word_Ct = 0
            NP_PRP = 0
            Keyword_window = 0
            Keyword_sink = 0
            Keyword_cookie = 0
            Keyword_curtain = 0
            Keyword_counter = 0
            Keyword_stool = 0
            
            words_only_dict = []
            
            
            #*************************************************
            #Info Units - Create synonyms
            #*************************************************
            
            boy_syns = {}
            for ss in wn.synsets('boy'):
                x = ss.lemma_names()
                for syn in x:
                    if syn not in boy_syns:
                        boy_syns[syn] = 0
                        
            for ss in wn.synsets('brother'):
                x = ss.lemma_names()
                for syn in x:
                    if syn not in boy_syns:
                        boy_syns[syn] = 0
            
            
            girl_syns = {}
            for ss in wn.synsets('girl'):
                x = ss.lemma_names()
                for syn in x:
                    if syn not in girl_syns:
                        girl_syns[syn] = 0
                        
            for ss in wn.synsets('sister'):
                x = ss.lemma_names()
                for syn in x:
                    if syn not in girl_syns:
                        girl_syns[syn] = 0
                    
            woman_syns = {}
            for ss in wn.synsets('woman'):
                x = ss.lemma_names()
                for syn in x:
                    if syn not in woman_syns:
                        woman_syns[syn] = 0
                    
            for ss in wn.synsets('mother'):
                x = ss.lemma_names()
                for syn in x:
                    if syn not in woman_syns:
                        woman_syns[syn] = 0
            
            for ss in wn.synsets('lady'):
                x = ss.lemma_names()
                for syn in x:
                    if syn not in woman_syns:
                        woman_syns[syn] = 0
                                             
            #************************************************
            #Reading in script
            #************************************************
            
            #Dx:  ProbableAD, PossibleAD, MCI, Memory, Vascular, Control
            
            with open(os.path.join(root, fname), 'r', encoding='utf8') as f:
                for line in f:
                    if line.startswith("@ID:"):
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
                        else:
                            dx = 'Other'
            #************************************************
            #Pre-processing
            #************************************************
                    if line.startswith("*"):
                        x = re.sub('[\s]', '_',line)
                        words = re.sub('[^\w]', '', x).split("_")
                        
                        i=0
                        words_len = len(words)             
                        while(i < words_len):
                            if(words[i]=='' or words[i].isdigit() or words[i]=='exc'):
            
                                del words[i]
                                i=i
                                words_len = len(words)
                            else:
                                i=i+1
                                words_len = len(words)
            
                        words_only = words[1:]
            
                        i=1
                        while(i < words_len):     
                            lmtzr.lemmatize(words[i])
                            i=i+1
            
                        pos_words = pos_tagger.tag(words_only)
                        parsed_out_pcfg  = reg_parser.parse(pos_words)
            
                        pre_parsed_out = dependency_parser.parse(words_only)
                        dep = pre_parsed_out.__next__()
                        parsed_out = list(dep.triples())
            
                        Script_Word_Ct += len(pos_words)                  
                        
                        i = 0
                        while(i < words_len-1):
                            tags = pos_words[i][1]
            
                            if(i<len(pos_words)-1 and tags == 'NP' and pos_words[i+1][1] == 'PRP'):
                                NP_PRP += 1
                            
                            i=i+1
                            for all in tags:
                                if tags in POS_COUNT:
                                    POS_COUNT[tags] += 1/len(pos_words)
                                else:
                                    POS_COUNT[tags] = 1/len(pos_words)
                                    
                        if(words[0] == 'PAR'):
                            i = 0
                            while(i < len(words_only)):
                                current_word = words_only[i]
                                if current_word in boy_syns:
                                    boy_syns[current_word] += 1
                                if current_word in girl_syns:
                                    girl_syns[current_word] += 1
                                if current_word in woman_syns:
                                    woman_syns[current_word] += 1
                                i=i+1
                                
                                
                        if(words[0] == 'PAR'):
                            i = 0
                            while(i < len(words_only)):
                                current_word = words_only[i]
                                if current_word in distinctWordCt:
                                    distinctWordCt[current_word] += 1
                                else:
                                    distinctWordCt[current_word] = 1
                                i=i+1            
                        i=1 
                        while(i < words_len):
                            Script_Letter_Ct += len(words[1:])
                            i=i+1
                     
                        if(words[0] == 'PAR'):
                            words_only_dict += [' '.join(words_only)]
                        
                        if(words[0] == 'PAR'):
                            PAR_Word_Ct += len(words[1:])
                            utt1 = ' '.join(words_only) #current line......
                            #print(utt1)
                        if(words[0] == 'PAR' and 'window' in words):
                            Keyword_window += 1
                        if(words[0] == 'PAR' and 'sink' in words):
                            Keyword_sink += 1
                        if(words[0] == 'PAR' and 'cookie' in words):
                            Keyword_cookie += 1
                        if(words[0] == 'PAR' and 'curtain' in words):
                            Keyword_curtain += 1
                        if(words[0] == 'PAR' and 'counter' in words):
                            Keyword_counter += 1
                        if(words[0] == 'PAR' and 'stool' in words):
                            Keyword_stool += 1
                        else:
                            INV_Word_Ct += len(words[1:])
            
            			
            INVtoPAR_word_ratio = INV_Word_Ct/PAR_Word_Ct          
            
            #Noun
            #ProNoun   
            #Pronoun:Noun Ratio
            Noun_Ct_Total = 0
            if('NN' in POS_COUNT): 
                Noun_Ct_Total += POS_COUNT['NN']
            if('NNS' in POS_COUNT): 
                Noun_Ct_Total += POS_COUNT['NNS']
            if('NNP' in POS_COUNT): 
                Noun_Ct_Total += POS_COUNT['NNP']
            
            ProNoun_Ct = 0
            if('PRP' in POS_COUNT):
                ProNoun_Ct += POS_COUNT['PRP']
            if('PRP$' in POS_COUNT):
                ProNoun_Ct += POS_COUNT['PRP$']
            
            ProNoun_Noun_Ratio = ProNoun_Ct/Noun_Ct_Total
            
            #Adverbs:
            #RB
            #RBR Adverb, comparative 
            #RBS Adverb, superlative
            
            Adverb_Ct_Total = 0
            if('RB' in POS_COUNT): 
                Adverb_Ct_Total += POS_COUNT['RB']
            if('RBR' in POS_COUNT): 
                Adverb_Ct_Total += POS_COUNT['RBR']
            if('RBS' in POS_COUNT):
                Adverb_Ct_Total += POS_COUNT['RBS']
            
            #Verb frequency:
            
            #VB   Verb, base form
            #VBD  Verb, past tense
            #VBG  Verb, gerund (a verb form which functions as a noun) or present participle
            #VBN  Verb, past participle
            #VBP  Verb, non-3rd person singular present
            #VBZ  Verb, 3rd person singular present
            
            Verb_Ct_Base = 0
            if('VB' in POS_COUNT):
                Verb_Ct_Base += POS_COUNT['VB']
            
            Verb_Ct_Past_T = 0
            if('VBD' in POS_COUNT):
                Verb_Ct_Past_T += POS_COUNT['VBD']
            
            Verb_Ct_Gerund = 0
            if('VBG' in POS_COUNT):
                Verb_Ct_Gerund += POS_COUNT['VBG']
            
            Verb_Ct_Past_P = 0
            if('VBN' in POS_COUNT): 
                Verb_Ct_Past_P += POS_COUNT['VBN']
            
            Verb_Ct_non3sing = 0
            if('VBP' in POS_COUNT):
                Verb_Ct_non3sing += POS_COUNT['VBP']
            
            Verb_Ct_3sing = 0
            if('VBZ' in POS_COUNT):
                Verb_Ct_3sing += POS_COUNT['VBZ']
            
            Verb_Ct_Total = Verb_Ct_Base+Verb_Ct_Past_T+Verb_Ct_Gerund+Verb_Ct_Past_P+Verb_Ct_non3sing+Verb_Ct_3sing
            Verb_Ct_Past = Verb_Ct_Past_T+Verb_Ct_Past_P
            
            #Word Length
            Avg_Word_Len = Script_Letter_Ct/Script_Word_Ct
             
            
            #*********************************************************************    
            #Cosine similarity to measure repeativeness
            
            #Using a bag-of-words, we measure the cosine distance between 
            #each pair of utterances in the session. We remove a short 
            #list of stopwords, after observing that utterances such as
            # He is standing on the stool and He is holding the cookie
            # could be considered relatively sim- ilar given the common
            # occurrences of he, is, and the.
            #********************************************************************* 
            
            #Need: Term-Doc Vector for each utterance
            #Therefore - need corpus to base Vector values on
            
            vect = TfidfVectorizer(min_df=1)
            
            i=0
            cos_total = 0
            while(i<len(words_only_dict)-1):
                utt0 = words_only_dict[i]
                utt1 = words_only_dict[i+1]
                
                tfidf2 = vect.fit_transform([utt0,utt1])
                
                sim_array2 = (tfidf2 * tfidf2.T).A
                i=i+1
                cos = sim_array2[1][0]
                cos_total += cos
            
            PAR_UTT_ct = len(words_only_dict)
            cos_avg = cos_total/PAR_UTT_ct
            
            #*********************************************************************
            #Honours Statistic
            
            #Honore’s Statistic (HS) attempts a deeper analysis by accounting for words 
            #that are only used once (V1), indicating a higher lexical richness.
            #Create key/val dictionary that has distinct words as KEY and freq count as VAL
            #Count how many times a work was used once.
            #Normalize over total number of words.
            #*********************************************************************
            
            wordsUsedOnce=0 
            disWordValsList = list(distinctWordCt.values())
            disWordKeysList = list(distinctWordCt.keys())
            i=0
            for i in disWordValsList:
                if(disWordValsList[i] == 1):
                    wordsUsedOnce += 1
                    i=i+1
            
            wordsOnceOverTotal = wordsUsedOnce/sum(disWordValsList) #Words used once/Total words - higher = higher lexical richness
            
            #*********************************************************************
            #Count words that are not in dictionary
            #*********************************************************************
            
            NID = 0
            i=0
            for all in disWordKeysList:
                if(disWordKeysList[i] not in wd.words()):
                    NID += 1
                    i=i+1
                else:
                    i=i+1
                    
            #*********************************************************************
            #Previous studies of AD narratives in the pictures description tasks have reported decreased information content.
            #Fraser measured this computationally by searching for relevant lexical items that point to 
            #each of the expected information units listen in Croisile et all.
            
            #I am only using the info units in the top 30 ish feaures
            #Ie. window,curtain,cookie,sink,girl,girl's action
            #*********************************************************************
            
            
            
            #************ --- INFO UNITS --- *************************************
            #Eg. Use the dependency structure to locate "fall" as the verb
            #and "boy" or "stool" as the subject
            
            
            #The amount of information mentioned in the texts was evaluated by means of 
            #a list of items constructed with the help of previous studies (Yorkston & 
            # Beukelman, 1980; Hier et al., 1985; Nicholas et al., 1985; Henderson et al., 1992). 
            #The list consisted of 23 information units in four key categories: subjects, places, 
            #objects, and actions. The three subjects were: the boy, the girl, and the woman.
            #The two places were the kitchen and the exterior seen through the window.
            #The eleven objects were: cookie, jar, stool, sink, plate, dishcloth, water,
            #window, cupboard, dishes, and curtains. The seven actions or facts were: boy taking 
            #or stealing, boy or stool falling, woman drying or washing dishes/plate, water 
            #overflowing or spilling, action performed by the girl, woman unconcerned by the overflowing, 
            #woman indifferent to the children.
            
            
            Subj_IU_Total = 0
            
            if sum(woman_syns.values()) != 0:
                Subj_IU_Total += 1
            if sum(boy_syns.values()) != 0:
                Subj_IU_Total += 1
            if sum(boy_syns.values()) != 0:
                Subj_IU_Total += 1
                
            Total_Keywords_Mentioned = Keyword_window+Keyword_sink+Keyword_cookie+Keyword_curtain+Keyword_counter+Keyword_stool
            
            series = pd.Series([dx
                                ,Adverb_Ct_Total
                                ,Avg_Word_Len
                                ,INV_Word_Ct
                                ,INVtoPAR_word_ratio
                                ,Keyword_cookie
                                ,Keyword_counter
                                ,Keyword_curtain
                                ,Keyword_sink
                                ,Keyword_stool
                                ,Keyword_window
                                ,Noun_Ct_Total
                                ,PAR_UTT_ct
                                ,PAR_Word_Ct
                                ,ProNoun_Ct
                                ,ProNoun_Ct
                                ,ProNoun_Noun_Ratio
                                ,Script_Letter_Ct
                                ,Script_Word_Ct,Subj_IU_Total
                                ,Total_Keywords_Mentioned
                                ,Verb_Ct_3sing
                                ,Verb_Ct_Base
                                ,Verb_Ct_Gerund
                                ,Verb_Ct_Past
                                ,Verb_Ct_Past_P
                                ,Verb_Ct_Past_T
                                ,Verb_Ct_Total
                                ,Verb_Ct_non3sing
                                ,cos_avg
                                ,wordsOnceOverTotal
                                ,wordsUsedOnce]
                                , name = fname)
            df = pd.concat([df, series], axis=1)
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
#    except: #handle other exceptions such as attribute errors
#        print("Unexpected error:", sys.exc_info()[0])

NIF_feats_names=['dx'
                                ,'Adverb_Ct_Total'
                                ,'Avg_Word_Len'
                                ,'INV_Word_Ct'
                                ,'INVtoPAR_word_ratio'
                                ,'Keyword_cookie'
                                ,'Keyword_counter'
                                ,'Keyword_curtain'
                                ,'Keyword_sink'
                                ,'Keyword_stool'
                                ,'Keyword_window'
                                ,'Noun_Ct_Total'
                                ,'PAR_UTT_ct'
                                ,'PAR_Word_Ct'
                                ,'ProNoun_Ct'
                                ,'ProNoun_Ct'
                                ,'ProNoun_Noun_Ratio'
                                ,'Script_Letter_Ct'
                                ,'Script_Word_Ct,Subj_IU_Total'
                                ,'Total_Keywords_Mentioned'
                                ,'Verb_Ct_3sing'
                                ,'Verb_Ct_Base'
                                ,'Verb_Ct_Gerund'
                                ,'Verb_Ct_Past'
                                ,'Verb_Ct_Past_P'
                                ,'Verb_Ct_Past_T'
                                ,'Verb_Ct_Total'
                                ,'Verb_Ct_non3sing'
                                ,'cos_avg'
                                ,'wordsOnceOverTotal'
                                ,'wordsUsedOnce']

df.to_pickle('/Users/clairekelleher/Desktop/Thesis/Data/nif_missing_feat_name.pkl')
df_missing = df.transpose()
