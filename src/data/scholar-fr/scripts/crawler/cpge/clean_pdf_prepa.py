import os 
import re
import pandas as pd 
import csv

duplicate_pattern = re.compile(r"_(e|c){1}[b-d]{1}\.pdf$")
sujet_pattern = re.compile(r"_(d|e|(e|d)a)\.pdf$")
correction_pattern = re.compile(r"_(c|ca)\.pdf$")
year_pattern = re.compile(r"(19|20)(9|0|1|2)[0-9]")
folder = ["../pdf/x-ens", "../pdf/ccinp", "../pdf/centrale", "../pdf/ccmp"]

csv_path = os.path.join("../datas/pdf_pairs.csv")

def clean_pdf(folder):
    files = os.listdir(folder)
    files_set = set(files)
    print(len(files))
    remove = []
    keep = []
    # remove all duplicates and if there is no correction 
    for file in files:
        if "(1)" in file:
            remove.append(file)
            continue
        if duplicate_pattern.search(file):
            remove.append(file)
            continue

        if not year_pattern.search(file):
            remove.append(file)
            continue
        if sujet_pattern.search(file):
            match = sujet_pattern.search(file).group()
            correction = file.replace(match, "_ca.pdf")
            if correction in files_set:
                keep.append([file,correction])
                files.remove(file)
                files.remove(correction)
                files_set.remove(file)
                files_set.remove(correction)
                continue
            continue
    
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Écrire l'en-tête uniquement si le fichier n'existe pas
        if not file_exists:
            writer.writerow(["Sujet", "Correction"])
        writer.writerows(keep)

    print(f"found {len(keep)}")



def main():
    for fold in folder:
        clean_pdf(fold)

if __name__=="__main__":
    main()