import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pyvis.network import Network
from collections import Counter
import networkx as nx
from graphrole import RecursiveFeatureExtractor
from collections import defaultdict


class TextGraph:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)  # Read the CSV that have food data, sports data, and diseases data
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.training_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.last_graph = None
        self.last_title = None
        self.training_graphs = []

    def preprocess_text(self, text):
        tokens = word_tokenize(text) 
        tokens = [token.lower() for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in self.stop_words] # Remove stopwords
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
        for topic, group in self.df.groupby('Topic'):
            self.training_data = pd.concat([self.training_data, group.head(12)], ignore_index=True)
            self.test_data = pd.concat([self.test_data, group.tail(3)], ignore_index=True)

    def process_training_data(self):
        for index, row in self.training_data.iterrows():
            title = row['Title']
            text = row['Text']

            if pd.isnull(text):
                continue
            tokenized_text = self.preprocess_text(text)
            graph = self.construct_graph(tokenized_text)
            self.last_graph = graph
            self.last_title = title
            print("Title:", title)
            print("Graph Nodes:", len(graph.nodes()))
            print("Graph Edges:", len(graph.edges()))
            print("\n")
            self.training_graphs.append(graph)

    def visualize_last_graph(self):
        if self.last_graph is not None:
            # Ensure the graph is directed
            if not self.last_graph.is_directed():
                self.last_graph = self.last_graph.to_directed()
            net = Network(notebook=True)
            net.from_nx(self.last_graph)
            # Tell PyVis to use the 'direction' attribute to determine the direction of the edges
            net.show("graph.html")
    def extract_role_features(self):
        role_features = []
        for graph in self.training_graphs:
            feature_extractor = RecursiveFeatureExtractor(graph)
            features = feature_extractor.extract_features()
            role_features.append(features)
            break
        return role_features

    def find_common_subgraphs(self):
        common_subgraphs = defaultdict(int)
        print("Training graphs:")
        for i, graph in enumerate(self.training_graphs):
            print(f"Graph {i}:")
            for component in nx.weakly_connected_components(graph):
                common_subgraphs[frozenset(component)] += 1
                print(f"  Component {common_subgraphs[frozenset(component)]}: {component}")
        common_subgraphs = {k: v for k, v in common_subgraphs.items() if v == len(self.training_graphs)}
        print(f"Number of common subgraphs: {len(common_subgraphs)}")
        return common_subgraphs




# Usage
tg = TextGraph("CombinedData.csv")
# divide the data into training and test sets
tg.divide_data()
# process the training data and construct graphs
tg.process_training_data()

# Extract common subgraphs from the training graphs
common_subgraphs = tg.find_common_subgraphs()

print("Number of common subgraphs:", len(common_subgraphs))

# Extract role-based features from the training graphs
# role_features = tg.extract_role_features()

# for i, features in enumerate(role_features):
#     print(f"Graph {i + 1} features:")
#     for node, role in features.items():
#         print(f"Node: {node}, Role: {role}")
#     print("\n") # Add a newline for readability

# Visualize the last graph



tg.visualize_last_graph()
