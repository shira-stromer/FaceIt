from faces_dataset import EmbeddingFacesDataSet
import argparse
import torch
from face_it_trainer import train_dbscan, train_dbscan_finetune
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Face-It trainer')
    parser.add_argument('--echo', type=int, default=5, help='echo')
    parser.add_argument('--use_wandb', default=False, action='store_true')

    parser.add_argument('--clustering_algorithm', type=str, default='dbscan', help='Supported algorithms: [dbscan, dbscan-ft, kmeans]')
    parser.add_argument('--min_cluster_samples', type=int, default=5)
    parser.add_argument('--dbscan_eps', type=float, default=0.7)

    

    return parser.parse_args()

def init_wandb(args):
    if args.use_wandb:
        wandb.init(
            project="FaceIt",
            entity='bronershira',
            config=args)

if __name__ == "__main__":
    print('hello world')
    args = parse_args()
    print(args)
    init_wandb(args)
    dataset = EmbeddingFacesDataSet()
    print(len(dataset))

    if args.clustering_algorithm == 'dbscan':
        train_dbscan(args.use_wandb, dataset, args.dbscan_eps, args.min_cluster_samples)
    elif args.clustering_algorithm == 'dbscan-ft':
        train_dbscan_finetune(args.use_wandb, dataset)
    else:
        raise NotImplementedError