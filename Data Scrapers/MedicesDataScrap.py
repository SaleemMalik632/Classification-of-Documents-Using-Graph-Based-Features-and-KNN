import requests
from bs4 import BeautifulSoup
import csv

url = "https://www.mayoclinic.org/diseases-conditions/index?letter=A"  # URL of the website


response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
main_content = soup.find(id="main-content") 
main_content = main_content.find(id="cmp-skip-to-main__content")
AllUl = main_content.find_all('ul')    
Alllinks = []
for ul in AllUl:
    links = ul.find_all('a')
    for link in links:
        Alllinks.append(link['href'])
        print(link['href'])


# get first 15 link 
news_data = []

for link in Alllinks:
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    header = ''
    headerRow = soup.find('div' , class_="row")  
    if headerRow is None:
        print("Header not found")
        continue
    header = headerRow.find('h1')
    print('Header:', header.text)
    main_content = soup.find(id="main-content")
    
    if main_content is None:
        print("Main content not found")
        continue

    news_paragraphs = main_content.find_all('p')
    
    # Initialize an empty string to store the news text
    news_text = ''
    for news_paragraph in news_paragraphs:
        news_text += news_paragraph.text
    news_data.append([header.text, news_text, len(news_text.split()), 'Diseases And Symptoms',link])
    if len(news_data) == 15:
        break



# Write the news data to a CSV file
with open('DiseasesAndSymptomsData.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "Text", "Word Count", "Topic","Link"])
    writer.writerows(news_data)




