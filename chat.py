import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import networkx as nx
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class TextGraph:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.training_data = pd.DataFrame()  # Initialize empty DataFrames
        self.test_data = pd.DataFrame()
        self.training_graphs = []
        self.labels = []

    def preprocess_text(self, text):
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def construct_graph(self, tokens):
        G = nx.DiGraph()
        for i in range(len(tokens) - 1):
            if not G.has_edge(tokens[i], tokens[i + 1]):
                G.add_edge(tokens[i], tokens[i + 1], weight=1, arrows='to')
            else:
                G.edges[tokens[i], tokens[i + 1]]['weight'] += 1
                G.edges[tokens[i], tokens[i + 1]]['arrows'] = 'to'
        return G

    def divide_data(self):
        self.training_data = pd.DataFrame()  # Initialize empty DataFrame for training data
        self.test_data = pd.DataFrame()  # Initialize empty DataFrame for test data
        for topic, group in self.df.groupby('Topic'):
            # Take the first 12 rows for training and the last 3 rows for testing
            self.training_data = pd.concat([self.training_data, group.head(12)], ignore_index=True)
            self.test_data = pd.concat([self.test_data, group.tail(3)], ignore_index=True)
            self.labels = group['Topic'].tolist()  # Assign labels to the labels attribute

    def preprocess_and_construct_graphs(self, data):
        graphs = []
        for text in data['Text']:
            if pd.isnull(text):
                continue
            tokenized_text = self.preprocess_text(text)
            graph = self.construct_graph(tokenized_text)
            graphs.append(graph)
        return graphs

    def adjacency_matrix(self, graph):
        return nx.adjacency_matrix(graph).todense()

    def calculate_adjacency_matrix_similarity(self, matrix1, matrix2):
        intersection = np.logical_and(matrix1, matrix2)
        union = np.logical_or(matrix1, matrix2)
        return np.sum(intersection) / np.sum(union)

    def extract_features(self, graphs):
        max_features = 110224  # Maximum number of features
        features = []
        for graph in graphs:
            adjacency_matrix = self.adjacency_matrix(graph)
            flattened_matrix = np.ravel(adjacency_matrix)
            if len(flattened_matrix) < max_features:
                # Pad feature with zeros if it's shorter than max_features
                padded_feature = np.pad(flattened_matrix, (0, max_features - len(flattened_matrix)), mode='constant')
                features.append(padded_feature)
            else:
                # Truncate feature if it's longer than max_features
                features.append(flattened_matrix[:max_features])
        return np.array(features)


    def knn_classification(self, X_train, y_train, X_test, y_test, k):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

# Usage
tg = TextGraph("CombinedData.csv")
tg.divide_data()
training_graphs = tg.preprocess_and_construct_graphs(tg.training_data)
test_graphs = tg.preprocess_and_construct_graphs(tg.test_data)

X_train = tg.extract_features(training_graphs)
X_test = tg.extract_features(test_graphs)
y_train = tg.labels
y_test = tg.labels

print("Number of features in X_train:", X_train.shape[1])
print("Number of features in X_test:", X_test.shape[1])


k = 3  # Number of neighbors for KNN
accuracy = tg.knn_classification(X_train, y_train, X_test, y_test, k)
print("Accuracy:", accuracy)
