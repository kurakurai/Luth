import wget
import csv

filter = "physique" # flter subjects to download

# This script downloads PDF files from a CSV file containing URLs.
def main():
    sujets_folder = 'pdf/sujets'
    corriges_folder = 'pdf/corriges'

    with open('pdf_urls.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        next(csvFile)
        for lines in csvFile:
            if '.' not in lines[0]:
                print(lines[0])

            if filter in lines[0]:
                if "corrige" in lines[0]:
                    folder = corriges_folder
                else:
                    folder = sujets_folder
                
                wget.download(lines[0], out=folder)

if __name__ == '__main__':
    main()