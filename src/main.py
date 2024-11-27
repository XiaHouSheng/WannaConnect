import requests
from bs4 import BeautifulSoup
from selenium import webdriver

url_root = "http://218.203.164.245:9980"
response = requests.get(url_root)
soup = BeautifulSoup(str(response.text),"lxml")
url = soup.select("html > frameset > frame:nth-child(2)")[0].get("src")
paramStr = url.split("=")[1]

options = webdriver.EdgeOptions()
# 忽略证书错误
options.add_argument('--ignore-certificate-errors')
service = webdriver.EdgeService("./driver/msedgedriver.exe")
driver = webdriver.Edge(options=options,service=service)
driver.get(url_root)
driver.implicitly_wait(100)
