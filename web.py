from bs4 import BeautifulSoup
import requests
from time import sleep
def create_url(brand, model, location):
    brand = brand.replace(' ', '%20')
    return 'https://www.autotrader.ca/cars/bmw/3%20series/on/toronto/?rcp=15&rcs=30&srt=3&prx=100&prv=Ontario&loc=M4Y0B9&hprc=True&wcp=True&sts=Used&inMarket=basicSearch'

if __name__ == '__main__':
