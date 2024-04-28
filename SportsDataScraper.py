import requests
from bs4 import BeautifulSoup
import csv

url = "https://www.bbc.com/sport/all-sports"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
main_content = soup.find(id="main-content")
category_divs = main_content.find_all('div', class_="ssrcss-1jx82ix-CategoryWrapper e1ef2m043")
Alllinks = []
for category_div in category_divs:
    stack_div = category_div.find('div', class_="ssrcss-wdw1q-Stack e1y4nx260")
    links = stack_div.find_all('a', class_="ssrcss-1wphvjl-Link ecc37rf0")
    for link in links:
        Alllinks.append(link['href'])
        print(link['href'])
new_url =  Alllinks[0]
response = requests.get('https://www.bbc.com/sport/american-football')
soup = BeautifulSoup(response.content, "html.parser")
main_content = soup.find(id="main-content")
SportsCardlinks = main_content.find_all('div', class_="ssrcss-1f3bvyz-Stack e1y4nx260")
links_list = []
for SportsCardlink in SportsCardlinks:
    links = SportsCardlink.find_all('a', class_="ssrcss-zmz0hi-PromoLink exn3ah91")
    for link in links:
        if "sport" in link['href']: 
            link['href'] = "https://www.bbc.com" + link['href']   
            links_list.append(link['href'])
            print(link['href'])
links_list = links_list[:-2]


news_data = []
for link in links_list:
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    main_content = soup.find(id="main-content")
    
    if main_content is None:
        print("Main content not found")
        continue
    news_heading = main_content.find(id="main-heading")
    
    if news_heading is None:
        print("News heading not found")
        continue
    
    print(news_heading.text) 
    news_paragraphs = main_content.find_all('div', class_="ssrcss-uf6wea-RichTextComponentWrapper ep2nwvo0")
    # Initialize an empty string to store the news text
    news_text = ''
    for news_paragraph in news_paragraphs:
        news_text += news_paragraph.text
        print(news_paragraph.text)
    news_data.append([news_heading.text, news_text, len(news_text.split()), 'Sport',link])
# Write the news data to a CSV file
with open('SportsData.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "Text", "Word Count", "Topic","Link"])
    writer.writerows(news_data)