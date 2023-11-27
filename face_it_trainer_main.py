from faces_dataset import EmbeddingFacesDataSet
import argparse
import torch
from dbscan_trainer import train_dbscan, train_dbscan_finetune
from kmeans_trainer import train_kmeans, train_kmeans_finetune
from utilities import create_directory
import shutil
import os
import wandb
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Face-It trainer')
    parser.add_argument('--random_seed', type=int, default=43)
    parser.add_argument('--use_wandb', default=True, action='store_true')

    parser.add_argument('--min_estimated_clusters', type=int, default=70)
    parser.add_argument('--max_estimated_clusters', type=int, default=200)

    parser.add_argument('--clustering_algorithm', type=str, default='kmeans-ft', help='Supported algorithms: [dbscan, dbscan-ft, kmeans, kmeans-ft]')
    
    # DBScan parameters
    parser.add_argument('--min_cluster_samples', type=int, default=5)
    parser.add_argument('--dbscan_eps', type=float, default=0.7)

    #KMeans parameters
    parser.add_argument('--k', type=int, default=122)
    parser.add_argument('--kmeans_init_method', type=str, default='random', help='Supported values: [kmeans++, random]')

    parser.add_argument('--clusters_output_dir', type=str, default=None) #'faces-labels'

    return parser.parse_args()

def init_wandb(args):
    if args.use_wandb:
        wandb.init(
            project="FaceIt",
            entity='bronershira',
            config=args)
        
def export_clusters(args, images_labels, cluster_labels):
    clusters_output_dir = args.clusters_output_dir
    if clusters_output_dir is None:
        return
    
    labels = {label:[images_labels[i] for i, l in enumerate(cluster_labels) if l == label] for label in set(cluster_labels)}
    create_directory(clusters_output_dir)
    for k, v in labels.items():
        create_directory(os.path.join(clusters_output_dir, str(k)), delete_if_exists=True)
        [shutil.copyfile(f, os.path.join(clusters_output_dir, str(k), f.split('/')[1])) for f in v]

if __name__ == "__main__":
    print('hello world')
    args = parse_args()
    print(args)
    init_wandb(args)
    np.random.seed(args.random_seed)
    dataset = EmbeddingFacesDataSet()
    print(len(dataset))

    clustering_algorithm = args.clustering_algorithm
    if clustering_algorithm == 'dbscan':
        image_labels, clusters_labels = train_dbscan(args.use_wandb, dataset, args.dbscan_eps, args.min_cluster_samples)
        export_clusters(image_labels, clusters_labels)
    elif clustering_algorithm == 'dbscan-ft':
        train_dbscan_finetune(args.use_wandb, dataset)
    elif clustering_algorithm == 'kmeans':
        image_labels, clusters_labels = train_kmeans(args.use_wandb, dataset, args.k)
        export_clusters(args, image_labels, clusters_labels)
    elif clustering_algorithm == 'kmeans-ft':
        train_kmeans_finetune(args.use_wandb, dataset, args.min_estimated_clusters, args.max_estimated_clusters)
    else:
        raise NotImplementedError