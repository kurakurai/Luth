import pandas as pd
import os


corrige = "../pdf/corriges"
sujets = "../pdf/sujets"

corrige_names = os.listdir(corrige)
sujet_names = os.listdir(sujets)
sujet_names = set(sujet_names)

pairs = []
for corrige in corrige_names: # find pairs
    sujet = corrige.replace("corrige", "sujet")
    if sujet in sujet_names:
        pairs.append([sujet, corrige])
    
    elif "-officiel" in corrige:
        sujet = corrige.replace("-officiel", "")
        if sujet in sujet_names:
            pairs.append([sujet, corrige])
    
    elif "-officiel" not in corrige:
        sujet = corrige.replace("corrige", "sujet-officiel")
        if sujet in sujet_names:
            pairs.append([sujet, corrige])
    
df = pd.DataFrame(pairs, columns=["sujet", "corrige"])
df.to_csv("../datas/pdf_pairs.csv", index=False)
