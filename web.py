from bs4 import BeautifulSoup
import requests
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def create_url(brand, model, location):
    brand, model = brand.replace(' ', '%20'), model.replace(' ', '%20')
    return f'https://www.autotrader.ca/cars/{brand}/{model}/?rcp=15&rcs=0&srt=3&prx=100&loc={location}&hprc=True' \
           f'&wcp=True&sts=Used&inMarket=basicSearch'


def scrap_data(brand, model, location):
    driver = webdriver.Chrome('C:/Users/rober/OneDrive/csc/car-price-web-scraping/chromedriver.exe')
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                            'Chrome/74.0.3729.169 Safari/537.36'}
    driver.get(create_url(brand, model, location))
    driver.implicitly_wait(100)
    driver.find_element_by_id('btnGotIt').click()

    SCROLL_PAUSE_TIME = 0.5
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Wait to load page
        sleep(SCROLL_PAUSE_TIME)
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    soup = BeautifulSoup(driver.page_source, 'lxml')
    # get all cars in this page
    all_tags = soup.find_all('div', 'col-xs-12 result-item')
    # trying to figure out the div structure
    # tag = all_tags[0].contents[3]
    # print(tag)
    for tag in all_tags:
        full_name = tag.contents[3].contents[3].findChildren('span')[0].text.strip()
        year = full_name[:4]
        # strip out the spec to get model name
        car_name = full_name[5:min([i for i in
                                    [full_name.find(','), full_name.find('|'), full_name.find('('),
                                     full_name.find('+'), full_name.find('*'), len(full_name)]
                                    if i > -1])]
        price = tag.contents[3].contents[5].findChildren('span')[0].text.strip()
        mileage = int(tag.contents[3].contents[3].find('div', class_='kms').text.split()[1].replace(',', ''))
        print(year, car_name, str(mileage) + 'km', price)


if __name__ == '__main__':
    scrap_data('bmw', '3 series', 'm4y0b9')