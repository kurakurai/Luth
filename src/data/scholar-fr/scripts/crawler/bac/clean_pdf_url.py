import pandas as pd

with open("../datas/pdf_urls copy.txt") as file:
    lines = [line.rstrip() for line in file]


corrige_urls = []
sujet_urls = set()

for pdf in lines:
    if "corrige" in pdf:
        corrige_urls.append(pdf)
    else : 
        sujet_urls.add(pdf)

download_urls = []
blacklist = [
    "espagnol",
    "anglais",
    "allemand",
    "chinois",
    "italien",
]


for corrige in corrige_urls: # find pairs and blacklist all the languages
    sujet = corrige.replace("corrige", "sujet")
    if any(lang in sujet for lang in blacklist):
        continue

    if sujet in sujet_urls:
        download_urls.append(sujet)
        download_urls.append(corrige)
    elif "-officiel" in corrige:
        sujet = corrige.replace("-officiel", "")
        if sujet in sujet_urls:
            download_urls.append(sujet)
            download_urls.append(corrige)
    elif "-officiel" not in corrige:
        sujet = corrige.replace("corrige", "sujet-officiel")
        if sujet in sujet_urls:
            download_urls.append(sujet)
            download_urls.append(corrige)        
    else:
        print(f"Missing sujet for {sujet}")
        print(f"Missing sujet for {corrige}")
        break

df = pd.DataFrame(download_urls, columns=["pdf_urls"])
df.to_csv("../datas/pdf_urls.csv", index=False)

print(f"download_urls: {len(download_urls)}")