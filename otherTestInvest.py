#************************** --Libraries-- ******************************

import os
import pandas as pd

df = pd.DataFrame([])
indir = '/Users/clairekelleher/Desktop/Thesis/Data/Pitt/Control/cookie'

for root, dirs, filenames in os.walk(indir):
    try:
        for fname in filenames[1:]:  
            fname = str(fname)
            
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
           
            feats_name = ['dx']
            series = pd.Series([dx], 
                                name = fname, index = feats_name)
            df = pd.concat([df, series], axis=1)
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

df = df.transpose()
groupByUpdate = pd.DataFrame(df)
gbu = groupByUpdate.groupby(df['dx']).size()

#dx - FLUENCY
#MCI            37
#Memory          3
#PossibleAD     15
#ProbableAD    172
#Vascular        5
#dtype: int64


#dx - RECALL
#MCI            41
#Memory          3
#PossibleAD     16
#ProbableAD    173
#Vascular        4

#dx - SENTENCE = "Can you give me a sentence using the word..."
#Control         1
#MCI            40
#Memory          3
#PossibleAD     16
#ProbableAD    172
#Vascular        5
#Non-Prob 64


#dx - COOKIE = "Describe everything you seeing this picture"
#MCI            43
#Memory          3
#PossibleAD     21
#ProbableAD    236
#Vascular        5

#dx - COOKIE CTRL
#Control    241
#dtype: int64

#Ideas - 
#Take out Co
#We know what NI feaures predict AD... Apply

#Look at different interactional features split by test....
# R