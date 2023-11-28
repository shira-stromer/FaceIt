from sklearn.cluster import DBSCAN
from faces_dataset import EmbeddingFacesDataSet
from sklearn.metrics import silhouette_score
import wandb
from collections import Counter
import tqdm


def train_dbscan(use_wandb:bool, dataset:EmbeddingFacesDataSet, eps, min_samples):
    images, image_labels = dataset.get_all()

    # Clustering with DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples) 
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
    return image_labels, labels

def train_dbscan_finetune(use_wandb:bool, dataset:EmbeddingFacesDataSet):
    images, image_labels = dataset.get_all()

    for eps in tqdm.tqdm(range(5, 100, 5), total=200):
        eps = eps / 100
        
        dbscan = DBSCAN(eps=eps, min_samples=10)  
        dbscan.fit(images.numpy())

        labels = [int(l) for l in dbscan.labels_]
        clusters_num = len(set(labels))

        score = float(silhouette_score(images.numpy(), labels)) if clusters_num > 1 else None

        dd = \
        {
            'algorithm': 'dbscan', 'eps': eps, 'min_samples':5, 'clusters_num': clusters_num, 
            'noisy_samples': len([l for l in labels if l == -1]), 
            'silhouette_score': score
        }

        wandb.log(dd) if use_wandb else print(dd)
