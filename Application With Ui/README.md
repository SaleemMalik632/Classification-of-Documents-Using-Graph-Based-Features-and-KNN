# Project Title: Graph-Based Text Classification

## Data Collection and Preparation:
- Collect or create 15 pages of text for each of the three assigned topics, ensuring each page contains approximately 300 words.
- Divide the dataset into a training set (12 pages per topic) and a test set (3 pages per topic).

## Graph Construction:
- Represent each page (document→500 words) as a directed graph where nodes represent unique terms (words), around 300 words, after preprocessing such as tokenization, stop-word removal, and stemming, and edges denote term relationships based on their sequence in the text.

## Feature Extraction via Common Subgraphs:
- Utilize frequent subgraph mining techniques to identify common subgraphs within the training set graphs. These common subgraphs will serve as features for classification, capturing the shared content across documents related to the same topic.

## Classification with KNN:
- Implement the KNN algorithm using a distance measure based on the maximal common subgraph (MCS) between document graphs. This involves computing the similarity between graphs by evaluating their shared structure, as indicated by the MCS.
- Classify test documents based on the majority class of their k-nearest neighbors in the feature space created by common subgraphs.

## Collaborators:
- Mustafa Riza

## Results:
- Achieved impressive accuracy and F1 scores in text classification.
- Demonstrated the effectiveness of graph-based approaches in uncovering hidden patterns and relationships in text data.

## GitHub Repository:
[GitHub Repository](https://github.com/SaleemMalik632/GT-Project.git)

## Working Video:
[Watch Working Video](./working%20video.mp4)

## Keywords:
Graph-Based Text Classification, Data Collection, Graph Construction, Feature Extraction, Classification, KNN Algorithm, Frequent Subgraph Mining, Text Analysis, Collaborative Work, Results.