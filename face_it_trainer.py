from sklearn.cluster import DBSCAN
from faces_dataset import EmbeddingFacesDataSet
import numpy as np
from sklearn.metrics import silhouette_score
import wandb

def train_dbscan(use_wandb:bool, dataset:EmbeddingFacesDataSet, eps, min_samples):
    # Assuming embeddings is an array of your face embeddings
    images, image_labels = dataset.get_all()

    # Clustering with DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # eps and min_samples need to be tuned
    dbscan.fit(images.numpy())

    # Cluster labels
    labels = dbscan.labels_
    for i, label in enumerate(labels):
        if label == -1:
            print(image_labels[i])

    print(f'Number of labels: {len(set(labels))}')


    # labels are the cluster labels from DBSCAN
    score = silhouette_score(images.numpy(), labels)
    print(score)
    pass

