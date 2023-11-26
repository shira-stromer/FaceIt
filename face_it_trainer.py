from sklearn.cluster import DBSCAN
from faces_dataset import EmbeddingFacesDataSet
import numpy as np
from sklearn.metrics import silhouette_score
import wandb
from collections import Counter
from utilities import create_directory
import shutil
import os
import tqdm


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
    export_clusters(image_labels, labels)
    pass

def export_clusters(images_labels, cluster_labels):
    labels = {label:[images_labels[i] for i, l in enumerate(cluster_labels) if l == label] for label in set(cluster_labels)}
    create_directory('faces-labels')
    for k, v in labels.items():
        create_directory(os.path.join('faces-labels', str(k)), delete_if_exists=True)
        [shutil.copyfile(f, os.path.join('faces-labels', str(k), f.split('/')[1])) for f in v]

def train_dbscan_finetune(use_wandb:bool, dataset:EmbeddingFacesDataSet):
    images, image_labels = dataset.get_all()

    for eps in tqdm.tqdm(range(5, 100, 5), total=200):
        eps = eps / 100
        #print(eps)
        
        dbscan = DBSCAN(eps=eps, min_samples=5)  # eps and min_samples need to be tuned
        dbscan.fit(images.numpy())

        labels = [int(l) for l in dbscan.labels_]
        clusters_num = len(set(labels))

        score = float(silhouette_score(images.numpy(), labels)) if clusters_num > 1 else None
        print(score)

        dd = \
        {
            'eps': eps, 'min_samples':5, 'clusters_num': clusters_num, 
            'noisy_samples': len([l for l in labels if l == -1]), #'clusters_counters': Counter(labels),
            'silhouette_score': score
        }

        #for key in dd.keys():
            #print(key, type(dd[key]))
        wandb.log(dd) if use_wandb else print(dd)
