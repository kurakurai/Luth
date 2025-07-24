from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, NoSuchElementException
import csv
import time

driver = webdriver.Firefox()
wait = WebDriverWait(driver, 10)

driver.get("https://prepas.org/index.php?module=Sujets")

def click_select2():
    for _ in range(5):  # on essaye plusieurs fois en cas de stale element
        try:
            select2_container = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "span.select2-selection")))
            select2_container.click()
            return True
        except StaleElementReferenceException:

            pass
    return False

if not click_select2():
    print("Impossible de cliquer sur la zone Select2")
    driver.quit()
    exit(1)

# Saisir le texte
try:
    select2_input = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "textarea.select2-search__field")))
    select2_input.send_keys("ccinp")
    select2_input.send_keys(Keys.ENTER)
except TimeoutException:
    print("Champ Select2 introuvable ou non cliquable")
    driver.quit()
    exit(1)


wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.dlink[href*='file=doc']")))


with open("pdf_urls.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["pdf_urls"])  # En-tête du CSV
    
    current_page = 1
    
    while True:
        print(f"Traitement de la page {current_page}...")
        

        try:
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.dlink[href*='file=doc']")))
        except TimeoutException:
            print(f"Aucun lien PDF trouvé sur la page {current_page}")
            break
        
        pdf_links = driver.find_elements(By.CSS_SELECTOR, "a.dlink[href*='file=doc']")
        

        for link in pdf_links:
            href = link.get_attribute("href")
            writer.writerow([href])
            print(f"Lien trouvé: {href}")
        
        print(f"Page {current_page}: {len(pdf_links)} liens trouvés")
        

        next_page = current_page + 1
        try:

            next_page_element = driver.find_element(By.CSS_SELECTOR, f"div#page{next_page}.page")
            

            if next_page_element.is_displayed() and next_page_element.is_enabled():
                print(f"Passage à la page {next_page}...")
                

                driver.execute_script("arguments[0].click();", next_page_element)
                

                time.sleep(2)
                
                try:
                    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.dlink[href*='file=doc']")))
                except TimeoutException:
                    print(f"Timeout en attendant les résultats de la page {next_page}")
                    break
                
                current_page = next_page
            else:
                print(f"Page {next_page} non accessible")
                break
                
        except NoSuchElementException:
            print(f"Aucune page {next_page} trouvée. Fin du crawling.")
            break
        except Exception as e:
            print(f"Erreur lors du passage à la page suivante: {e}")
            break

print("Crawling terminé. Tous les liens ont été sauvegardés dans pdf_urls.csv")
driver.quit()