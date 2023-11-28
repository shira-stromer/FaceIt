<h1 align="center">
  FaceIt 😊
</h1>

<p align="center">
  <strong>Unsupervised Facial Clustering in Image Data</strong>
</p>

<p align="left">
This project is designed with the goal of implementing unsupervised facial clustering within image datasets. Our primary objective is to efficiently organize photographs from events by identifying and grouping images based on the faces of individuals captured in them. Utilizing advanced techniques in unsupervised learning, the project focuses on the nuanced task of detecting and clustering facial features, thereby facilitating a streamlined and intuitive organization of event photos. This approach not only enhances the accessibility of specific images within large datasets but also brings a novel perspective to photo management and organization driven by facial recognition technology.
</p>

## Project Structure

Face-It consists of a pipeline that processes a directory of photos, extracts faces, and creates a dataset of face embeddings. It then applies unsupervised clustering algorithms to group these faces.

- Data Preparation - [data_preparation_main.py](https://github.com/shirabroner/FaceIt/blob/main/data_preparation_main.py) 

- Model Training - [face_it_trainer_main.py](https://github.com/shirabroner/FaceIt/blob/main/face_it_trainer_main.py). 

## Data Preparation

Usage:
`python data_preparation_main.py --faces_images_dir [path_to_faces] --faces_embeddings_pt [output_embeddings_file]`

This process involves:

- Detecting faces in each image using MTCNN.
- Evaluating the quality of detected faces based on size and blurriness.
- Saving the high-quality face images in the specified directory.
- Converting Faces to Embeddings
- After extracting and filtering the face images, the script converts them into embeddings using a pre-trained InceptionResnetV1 model

#### Weights & Biases Integration
This script is integrated with Weights & Biases (wandb) for logging and tracking. To use wandb, set up your wandb account and project, then pass the project and entity names to the script.

#### Arguments
- --images_dir: Directory containing the original images.
- --faces_images_dir: Directory to store extracted face images.
- --faces_embeddings_pt: Path to save the generated embeddings.
- --use_wandb: Flag to enable wandb logging.
- --wandb_project: Your wandb project name.
- --wandb_entity: Your wandb entity name.
- --device: The device to use for computations ('cpu', 'cuda', 'mps').

## Modeling
Usage:
`python face_it_trainer_main.py --your_arguments`

#### Arguments
- --random_seed: Seed for random number generation.
- --use_wandb: Flag to enable Weights & Biases logging.
- --wandb_project: Project name for Weights & Biases.
- --wandb_entity: Entity name for Weights & Biases.
- --faces_embeddings_pt: Path to the .pt file generated after running the data preparation script.
- --min_estimated_clusters and --max_estimated_clusters: Estimated range for the number of clusters.
- Clustering algorithm parameters (e.g., --dbscan_eps, --min_cluster_samples, --k).
- --clusters_output_dir: Directory to store output clusters.

#### Clustering Algorithms
Face-It supports multiple clustering algorithms:

- DBSCAN and DBSCAN Finetune (dbscan, dbscan-ft)
- KMeans and KMeans Finetune (kmeans, kmeans-ft)

Specify the algorithm using the --clustering_algorithm argument.

#### Exporting Clusters
The clustered faces are exported to the specified output directory, organized by cluster labels.

## License

This project is licensed under the MIT License

## Wandb Graphs
![image](https://github.com/shirabroner/FaceIt/assets/33096214/5c90437e-b8d8-411b-a1c0-c0956fa805af)





