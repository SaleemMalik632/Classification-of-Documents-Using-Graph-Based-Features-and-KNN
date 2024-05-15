import pandas as pd

# Read the CSV files into dataframes
df1 = pd.read_csv('DiseasesAndSymptomsData.csv', encoding='latin1')
df2 = pd.read_csv('SportsData.csv' , encoding='latin1')
df3 = pd.read_csv('FooDData.csv', encoding='latin1')

# Concatenate the dataframes
df = pd.concat([df1, df2, df3])

# Write the combined dataframe to a new CSV file
df.to_csv('CombinedData.csv', index=False)











# Here's a step-by-step approach you can follow:

# - Load your scraped data from the file, including normal words, topic labels, word count, and blog title.
# - Preprocess the text data using tokenization, stop-word removal, and stemming.
# - Construct graphs for each document using the preprocessed text.
# - Extract features from the graphs using frequent subgraph mining techniques.
# - Implement the KNN algorithm using a distance measure based on maximal common subgraph (MCS) similarity.
# - Train your KNN model using the training set.
# - Classify test documents based on their nearest neighbors in the feature space created by common subgraphs.
# # - Evaluate the performance of your model using the test set.
