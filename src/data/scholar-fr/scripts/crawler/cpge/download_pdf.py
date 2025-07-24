import csv
import wget
import os
import urllib.error

def main():
    folder = '../pdf/ccmp'

    os.makedirs(folder, exist_ok=True)

    with open('/home/gad/kurakura/prepa_fr/datas/pdf_urls_ccmp.csv', mode='r') as file:
        csvFile = csv.reader(file)
        next(csvFile)  # Skip header if exists

        for i, line in enumerate(csvFile):
            url = line[0].strip()

            if not url:
                print(f"Ligne {i + 2} vide, ignorée.")
                continue

            try:
                print(f"Téléchargement de : {url}")
                wget.download(url, out=folder)
                print("Fait")
            except urllib.error.HTTPError as e:
                print(f"Erreur HTTP ({e.code}) pour {url}")
            except Exception as e:
                print(f"Autre erreur pour {url} : {e}")

if __name__ == '__main__':
    main()
