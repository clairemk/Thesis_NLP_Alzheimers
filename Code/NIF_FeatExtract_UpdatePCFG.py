
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
import os
from nltk.parse import stanford
from nltk.tree import ParentedTree


df = pd.DataFrame([])

#************************** --STANFORD POS TAGGER-- ******************************
_stanford_url = 'https://nlp.stanford.edu/software/lex-parser.shtml'
jar = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-postagger-2015-12-09/stanford-postagger.jar'
model = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-postagger-2015-12-09/models/english-left3words-distsim.tagger'

pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')
stanford_dir = pos_tagger._stanford_jar.rpartition('/')[0]
stanford_jars = find_jars_within_path(stanford_dir)
pos_tagger._stanford_jar = ':'.join(stanford_jars)

#************************** --STANFORD PARSER-- **********************************
_MAIN_CLASS = 'edu.stanford.nlp.parser.lexparser.LexicalizedParser'
os.environ['STANFORD_PARSER'] = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/jars'
os.environ['STANFORD_MODELS'] = '/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/jars'
os.putenv("CLASSPATH", "/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar")
java_path = "C:/Program Files/Java/jdk1.7.0_11/bin/java.exe"
os.environ['JAVAHOME'] = java_path

path_to_jar_p = "/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/jars/stanford-parser.jar"
path_to_models_jar_p = "/Users/clairekelleher/Desktop/Thesis/Fromdesktop/stanford-parser-full-2017-06-09/jars/stanford-parser-3.8.0-models.jar"
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar_p, path_to_models_jar=path_to_models_jar_p)
pcfg_parser = stanford.StanfordParser(path_to_jar=path_to_jar_p, path_to_models_jar=path_to_models_jar_p)

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


#        for subsubtree in subtree.subtrees():
#            if subsubtree.label() == "PRP":
#                print(subsubtree)
#def file_len(fname):
#    with open(fname) as f:
#        for i, l in enumerate(f):
#            pass
#        return i + 1
#********************** --Create read in fn-- ******************
script_no=1
#fname = "002-0.cex"
indir = '/Users/clairekelleher/Desktop/Thesis/Data/PItt_cookie_all_indf_IF'

for root, dirs, filenames in os.walk(indir):
    try:
        for fname in filenames[1:]:  
            fname = str(fname)
            #************************************************
            #Reading in script
            #************************************************
            
            #Dx:  ProbableAD, PossibleAD, MCI, Memory, Vascular, Control
            Total_PAR_Utts = 0
            NP_to_PRP = 0
            NP_to_DT_NN = 0
            ADVP_to_RB = 0
            ROOT_to_FRAG = 0    
            VP_to_AUX_VP = 0
            VP_VBG = 0
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
                        
                    if line.startswith("*PAR"):
                            
                            Total_PAR_Utts += 1
                            words = line.split(" ")
                            words[0] = words[0].replace("*PAR:\t","")
                            del words[len(words)-1]
                            del words[len(words)-1]
                            
                            for index, word in enumerate(words):
                                if words[index].startswith("[") or words[index].startswith(".") or words[index].startswith("+"):
                                    del words[index]
                                else:
                                    for char in words[index]:
                                        if char.isalpha() is False:
                                            del char
                            words_only = words
#                            print(words_only)
                            pos_words = pos_tagger.tag(words_only)
                            parsed_out_pcfg  = pcfg_parser.tagged_parse(pos_words)
                            
                            for all in parsed_out_pcfg:
                                ptree = ParentedTree.fromstring(str(all))
                                
                            for subtree in ptree.subtrees():
                                if subtree.label() == "NP":
                                    for all in subtree.subtrees():
                                        if all.label() == "PRP":
                                            NP_to_PRP += 1
#                            ADVP -> RB
                                if subtree.label() == "ADVP":
                                    for all in subtree.subtrees():
                                        if all.label() == "RB":
                                            ADVP_to_RB += 1
#                           NP -> DT NN
                                if subtree.label() == "NP":
                                    for subsubtree in subtree.subtrees():
                                        if subsubtree.label() == "DT":
                                            for all in subtree.subtrees():
                                                if all.label() == "NN" or all.label() == "NNS":     
                                                    NP_to_DT_NN += 1

                                if subtree.label() == "ROOT":
                                    for all in subtree.subtrees():
                                        if all.label() == "FRAG":
                                            ROOT_to_FRAG += 1
                                            
                                if subtree.label() == "VP":
                                    for all in subtree.subtrees():
                                        if all.label() == "AUX" or all.label() == "VP":
                                            VP_to_AUX_VP += 1
                                            
                                if subtree.label() == "VP":
                                        for all in subtree.subtrees():
                                            if all.label() ==  "VBP":
                                                VP_VBG += 1

            NP_to_PRP_norm = NP_to_PRP/Total_PAR_Utts
            ADVP_to_RB_norm = ADVP_to_RB/Total_PAR_Utts
            NP_to_DT_NN_norm = NP_to_DT_NN/Total_PAR_Utts
            ROOT_to_FRAG_norm = ROOT_to_FRAG/Total_PAR_Utts
            VP_to_AUX_VP_norm = VP_to_AUX_VP/Total_PAR_Utts
            VP_VBG_norm = VP_VBG/Total_PAR_Utts
            feats_name = ['dx'
                          ,'NP_to_PRP_norm'
                          ,'NP_to_PRP'
                          ,'ADVP_to_RB_norm'
                          ,'ADVP_to_RB'
                          ,'NP_to_DT_NN_norm'
                          ,'NP_to_DT_NN'
                          ,'ROOT_to_FRAG_norm'
                          ,'ROOT_to_FRAG'
                          ,'VP_to_AUX_VP_norm'
                          ,'VP_to_AUX_VP'
                          ,'VP_VBG_norm'
                          ,'VP_VBG']
            series = pd.Series([dx
                                ,NP_to_PRP_norm
                                ,NP_to_PRP
                                ,ADVP_to_RB_norm
                                ,ADVP_to_RB
                                ,NP_to_DT_NN_norm
                                ,NP_to_DT_NN
                                ,ROOT_to_FRAG_norm
                                ,ROOT_to_FRAG
                                ,VP_to_AUX_VP_norm
                                ,VP_to_AUX_VP
                                ,VP_VBG_norm
                                ,VP_VBG]
                                , name = fname, index = feats_name)
            df = pd.concat([df, series], axis=1)
            print(script_no)
            print(fname)
            script_no=script_no+1
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        
df.to_pickle('/Users/clairekelleher/Desktop/Thesis/Data/new_PARSE_all.pkl')
