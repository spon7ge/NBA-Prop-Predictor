import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pyautogui as p
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from Supplier import Supplier

class PrizePicks_Scraper():
    def __init__(self):
        supplier = Supplier()
        self.lines = []
        self.directory = supplier.getDirectory()
        self.getJSON()
        self.loadJSON()
    
    def getJSON(self):
        url = "https://api.prizepicks.com/projections?league_id=7"
        driver = webdriver.Firefox()
        driver.get(url)

        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.ID, "json-tab")))
        save_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn.save")))
        save_button.click()
        time.sleep(2)
        p.press("enter")
        time.sleep(.5)
        p.press("left")
        time.sleep(.5)
        p.press("enter")
        time.sleep(2)

        driver.quit()

    def loadJSON(self):
        with open(self.directory, 'r') as file:
            data = json.load(file)

        player_names = {}
        for elem in data['included']:
            if elem['type'] == 'new_player':
                player_names[elem['id']] = elem['attributes']['name']

        player_projections = []
        for proj in data['data']:
            if proj['type'] == 'projection':
                player_id = proj['relationships']['new_player']['data']['id']
                player_name = player_names.get(player_id)

                line_score = proj['attributes']['line_score']
                stat_type = proj['attributes']['stat_type']
                flash_sale = proj['attributes']['odds_type']

                start_time = proj['attributes']['start_time']
                dt = datetime.fromisoformat(start_time)
                pacific_time = dt.astimezone(ZoneInfo("America/Los_Angeles"))
                month = pacific_time.strftime("%b")
                day = pacific_time.strftime("%d").lstrip('0')
                formatted_date = pacific_time.strftime(f"{month}-{day}-%Y %I:%M %p")
                
                player_projections.append((player_name,stat_type,line_score, flash_sale, formatted_date))
        self.lines = player_projections