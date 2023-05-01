import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class KMeansOnDigits:
    
    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
    def load_dataset(self):
        self.digits = load_digits()
        
    def predict(self):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.clusters = kmeans.fit_predict(self.digits.data)
        
    def get_labels(self):
        self.labels = np.zeros_like(self.clusters)
        for i in range(self.n_clusters):
            mask = (self.clusters == i)
            self.labels[mask] = mode(self.digits.target[mask])[0]
        
    def calc_accuracy(self):
        self.accuracy = round(accuracy_score(self.digits.target, self.labels), 2)
        
    def confusion_matrix(self):
        self.mat = confusion_matrix(self.digits.target, self.labels)
        sns.heatmap(self.mat.T, square=True, annot=True, fmt='d', cbar=False)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
