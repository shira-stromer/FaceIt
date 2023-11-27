from sklearn.cluster import DBSCAN
from faces_dataset import EmbeddingFacesDataSet
import numpy as np
from sklearn.metrics import silhouette_score
import wandb
from collections import Counter
from sklearn.cluster import KMeans
import tqdm


#TODO: More parameters to fine tune

def train_kmeans(use_wandb:bool, dataset:EmbeddingFacesDataSet, k):
    # Assuming images is an array of your face embeddings
    images, image_labels = dataset.get_all()

    # Clustering with KMeans
    kmeans = KMeans(n_clusters=k, n_init=10)  
    kmeans.fit(images.numpy())

    # Cluster labels
    labels = kmeans.labels_
    for i, label in enumerate(labels):
        if label == -1:
            print(image_labels[i])

    print(f'Number of labels: {len(set(labels))}')

    score = silhouette_score(images.numpy(), labels)
    print(score)
    return image_labels, labels

def train_kmeans_finetune(use_wandb:bool, dataset:EmbeddingFacesDataSet):
    images, _ = dataset.get_all()

    range_n_clusters = range(2, 150) 
    numpy_array = images.numpy()

    # Elbow Method
    wcss = []
    for k in tqdm.tqdm(range_n_clusters, desc='Processing', total=len(range_n_clusters)) :
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(numpy_array)
        labels = kmeans.labels_
        wcss.append(kmeans.inertia_)

        lables_counter = Counter(labels)
        clusters_num = len(lables_counter)
        score = float(silhouette_score(images.numpy(), labels)) if clusters_num > 1 else None

        dd = \
        {
            'algorithm': 'kmeans', 'k': k, 'min_samples':lables_counter.most_common()[::-1][0][1], 'clusters_num': clusters_num, 
            'noisy_samples': -1, #'clusters_counters': Counter(labels),
            'wcs': kmeans.inertia_, 'silhouette_score': score
        }
        wandb.log(dd) if use_wandb else print(dd)

    print(wcss)
