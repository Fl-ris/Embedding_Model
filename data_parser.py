########################################
## Parser voor PubMed .xml bestanden  ##
## Floris Menninga                    ##
## Datum: 08-01-2026                  ##
########################################


import os
import glob
from bs4 import BeautifulSoup
from tqdm import tqdm


class xml_parser:


    def __init__(self, input_dir, output_dir, keywords):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.keywords = keywords



    def parse_pmc_xml(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'xml')

            # Titel van artikel:
            title_tag = soup.find('article-title')

            title_text = title_tag.get_text()


            # Filter op keywords:
            if self.keywords:
                if not any(k.lower() in title_text.lower() for k in self.keywords):
                    return None # Gebruik artikel niet als geen van de keywords er in voor komen.

            # De "fulltext" van het artikel:
            body_tag = soup.find('body')
            body_text = ""
            
            if body_tag:
                # Geen tabellen etc. 
                paragraphs = body_tag.find_all('p')
                
                # Join paragrafen:
                body_text = " ".join([p.get_text(strip=True) for p in paragraphs])

            return f"{title_text}. {body_text}"

        except Exception as e:
            print(f"Error {self.input_dir}: {e}")
            return None
        

    def run(self):
        # Lijst van alle .xml bestanden in de gegeven dir:
        xml_files = glob.glob(os.path.join(self.input_dir, "*.xml"))
        print(f"{len(xml_files)} XML bestanden...")

        with open(self.output_dir, 'w', encoding='utf-8') as out_f:
            
            count = 0
            # tqdm voor voortgangsbalk:
            for file_path in tqdm(xml_files):
                article_text = self.parse_pmc_xml(file_path)
                
                if article_text:
                    # Schrijf per regel een artikel:
                    out_f.write(article_text + "\n")
                    count += 1

        print(f"Voltooid...")
        print(f"{count} Artikelen met deze zoekcriteria opgeslagen in: {self.output_dir}")
