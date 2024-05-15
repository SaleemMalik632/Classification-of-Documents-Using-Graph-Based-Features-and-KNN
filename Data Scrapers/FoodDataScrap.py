import requests
from bs4 import BeautifulSoup
import csv

url = "https://www.foodnetwork.com/recipes"  # URL of the website


response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
main_content = soup.find(id="site") 
category_divs = main_content.find('div', class_="articleBody parsys")
AllCardlinks = category_divs.find_all('div', class_="editorialPromo capsule section") 
Alllinks = []
for Cardlink in AllCardlinks:
    links = Cardlink.find_all('a', class_="m-MediaBlock__m-ResponsiveImage--Link") 
    for link in links:
        link['href'] = "https:" + link['href']
        if "recipes" in link['href']:
            Alllinks.append(link['href'])
            print(link['href'])

news_data = []

linkCount = 0
for link in Alllinks:
    linkCount += 1
    print(linkCount)
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    main_content = soup.find(id="site")
    
    if main_content is None:
        continue

    MainHeading = main_content.find('h1', class_="o-AssetTitle__a-Headline") 
    if MainHeading is None:
        continue

    RescipesContant = main_content.find('div', class_="galad-container") 
    if RescipesContant is None:
        continue

    news_paragraphs = RescipesContant.find('div', class_="o-AssetDescription__a-Description")
    if news_paragraphs is None:
        continue

    news_text = ''
    news_text += news_paragraphs.text

    Allrespices = main_content.find('div', class_="slides")
    if Allrespices is None:
        continue

    Allrespices = Allrespices.find_all('div', class_="slide")
    for respices in Allrespices:
        respicesText = respices.find('div', class_="slide-caption") 
        news_text += respicesText.text
        print(respicesText.text)
        if respicesText is None:
            continue
    news_data.append([MainHeading.text, news_text, len(news_text.split()), 'Food',link])



# Write the news data to a CSV file
with open('FooDData.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "Text", "Word Count", "Topic","Link"])
    writer.writerows(news_data) 


# id="__next"

# id="cmp-skip-to-main__content"    