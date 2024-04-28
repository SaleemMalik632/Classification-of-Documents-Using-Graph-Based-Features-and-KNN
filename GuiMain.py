import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from Mainwindow import Ui_MainWindow
import recources_rc
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import networkx as nx
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import networkx as nx
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import os
import networkx as nx 
from pyvis.network import Network


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.btnLoadFile.clicked.connect(self.load_file)  # Connect the load_file method to the clicked signal of the Load button
        self.last_graph = None
    
    def showgraph(self):
        # net = Network(notebook=True)
        # net.from_nx(self.last_graph)
        # net.show_buttons(filter_=['physics'])  
        # net.save_graph("graph.html")
        self.webview = QWebEngineView()
        file_path = os.path.abspath('graph.html')
        url = QUrl.fromLocalFile(file_path)
        self.webview.load(url) 
        self.webview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.webview.setZoomFactor(2)
        while self.webfram.layout().count():
            child = self.webfram.layout().takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.webfram.layout().addWidget(self.webview)

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.lineEditShowfile.setText(file_name)
            self.df = pd.read_csv(file_name)
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.graphs = []
            self.training_data = pd.DataFrame()
            self.test_data = pd.DataFrame()
            self.accuracy_list = []
            self.divide_data()
            self.train_knn()
            self.test_knn()

    def preprocess_text(self, text):
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def construct_graph(self, tokens):
        G = nx.Graph()  # Change to directed graph
        for i in range(len(tokens) - 1):
            G.add_edge(tokens[i], tokens[i + 1])
        return G

    def divide_data(self):
        self.training_data = self.df.groupby('Topic').head(15)
        self.test_data = self.df.groupby('Topic').tail(5)

    def process_data(self, data):
        graphs = []
        for index, row in data.iterrows():
            text = row['Text']
            if pd.isnull(text):
                continue
            tokenized_text = self.preprocess_text(text)
            graph = self.construct_graph(tokenized_text)
            self.last_graph = graph
            graphs.append(graph)
        return graphs

    def extract_common_subgraphs(self, graphs):
        common_subgraphs = defaultdict(int)
        for graph in graphs:
            for component in nx.connected_components(graph):
                common_subgraphs[frozenset(component)] += 1
        return common_subgraphs

    def extract_features(self, data):
        features = []
        labels = []
        graphs = self.process_data(data)
        common_subgraphs = self.extract_common_subgraphs(graphs)
        for i, graph in enumerate(graphs):
            graph_features = [1 if frozenset(component) in common_subgraphs else 0 for component in nx.connected_components(graph)]
            features.append(graph_features)
            labels.append(data.iloc[i]['Topic'])
        return features, labels 

    def train_knn(self):
        X_train, y_train = self.extract_features(self.training_data)
        self.knn = KNeighborsClassifier(n_neighbors=2)
        self.knn.fit(X_train, y_train)
        for k in range(1, 11):
            self.knn = KNeighborsClassifier(n_neighbors=k)
            self.knn.fit(X_train, y_train)
            X_test, y_test = self.extract_features(self.test_data)
            y_pred = self.knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.accuracy_list.append(accuracy)

    def plot_accuracy(self):
        fig, ax = plt.subplots(figsize=(4, 3.5), facecolor='#2F4F7F')  
        ax.plot(range(1, 11), self.accuracy_list, color='#007bff', linewidth=2, marker='o')  
        ax.set_title("Accuracy Over Different Number of Neighbors", fontsize=12, color='white')  
        ax.set_xlabel("Number of Neighbors", fontsize=10, color='white')  
        ax.set_ylabel("Accuracy", fontsize=10, color='white')  
        ax.grid(True)  
        ax.tick_params(axis='both', which='major', labelsize=10, labelcolor='white')  
        fig.canvas.draw()
        img = QtGui.QPixmap.fromImage(QtGui.QImage(bytearray(fig.canvas.buffer_rgba()), fig.canvas.width(), fig.canvas.height(), QtGui.QImage.Format_RGBA8888))
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(img)
        self.graphicsView_for_accuracy.setScene(scene)
        self.graphicsView_for_accuracy.setStyleSheet("background-color: white; border-radius: 30px;")
        self.graphicsView_for_accuracy.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView_for_accuracy.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    def confusion_matrix_to_image(self, cm):
        fig, ax = plt.subplots(figsize=(6, 3.5))  
        ax.imshow(cm, interpolation='nearest', cmap='Blues')  
        ax.set_title("Confusion Matrix", fontsize=12, color='white')  
        fig.canvas.draw()
        img = QtGui.QPixmap.fromImage(QtGui.QImage(bytearray(fig.canvas.buffer_rgba()), fig.canvas.width(), fig.canvas.height(), QtGui.QImage.Format_RGBA8888))
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(img) 
        text = QtWidgets.QGraphicsTextItem("Confusion Matrix")
        text.setFont(QtGui.QFont("Arial", 12))
        text.setPos(10, 10)
        scene.addItem(text)
        self.graphicsView_for_matrix.setScene(scene)
        self.graphicsView_for_matrix.setStyleSheet("background-color: white; border-radius: 30px;")
        self.graphicsView_for_matrix.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView_for_matrix.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    


    def test_knn(self):
        X_test, y_test = self.extract_features(self.test_data)
        y_pred = self.knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')  
        f1 = f1_score(y_test, y_pred, average='weighted')
        self.AccuracyLabel.setText(f"Accuracy: {accuracy*100:.2f}%")
        self.PrecisionLabel.setText(f"Precision: {precision*100:.2f}%")
        self.RecallLabel.setText(f"Recall: {recall*100:.2f}%")
        self.F1ScoreLabel.setText(f"F1-score: {f1*100:.2f}%")
        self.plot_accuracy()
        cm = confusion_matrix(y_test, y_pred)
        self.confusion_matrix_to_image(cm)
        self.showgraph()

        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow() 
    window.show()
    sys.exit(app.exec_())
