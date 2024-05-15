import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("CombinedData.csv")
X_train, X_test, y_train, y_test = train_test_split(df["Text"], df["Topic"], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = RandomForestClassifier(n_estimators=100) # 100 decision trees
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)

print("Predicted Labels:", y_pred, "\nTrue Labels:", y_test) 

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
# Plot a confusion matrix

confusion = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(confusion, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()