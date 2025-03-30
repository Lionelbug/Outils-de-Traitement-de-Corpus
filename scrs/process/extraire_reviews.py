import requests
from bs4 import BeautifulSoup
import pandas as pd

data = []

headers = {'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3'} 
for i in range(1, 6):
    url = 'http://steamcommunity.com/app/433850/homecontent/?userreviewsoffset=' + str(10 * (i - 1)) + '&p=' + str(i) + '&workshopitemspage=' + str(i) + '&readytouseitemspage=' + str(i) + '&mtxitemspage=' + str(i) + '&itemspage=' + str(i) + '&screenshotspage=' + str(i) + '&videospage=' + str(i) + '&artpage=' + str(i) + '&allguidepage=' + str(i) + '&webguidepage=' + str(i) + '&integratedguidepage=' + str(i) + '&discussionspage=' + str(i) + '&numperpage=10&browsefilter=toprated&browsefilter=toprated&appid=433850&appHubSubSection=10&l=schinese&filterLanguage=default&searchText=&forceanon=1'
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, 'html.parser')  
    reviews = soup.find_all('div', {'class': 'apphub_Card'})    
    for review in reviews:
        tag = review.find('div', {'class': 'title'}).text.strip()
        hour = review.find('div', {'class': 'hours'}).text.split(' ')[0]
        comment = review.find('div', {'class': 'apphub_CardTextContent'}).text.strip()
        data.append({'Tag': tag, 'Hours': hour, 'Comment': comment})
df = pd.DataFrame(data)
df.to_csv('../data/raw/steam_reviews.csv', index=False, encoding='utf-8-sig')
print("sauvegarder：steam_reviews.csv")