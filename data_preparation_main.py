import importlib
import utilities
importlib.reload(utilities)
from utilities import get_files_from_dir
from facenet_pytorch import InceptionResnetV1
from PIL import Image, ImageDraw
import torch
import os
from torchvision import transforms
from facenet_pytorch import MTCNN, extract_face
import tqdm
importlib.reload(tqdm)
import cv2
import numpy as np
import argparse
import wandb
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Face-It data preparator')
    parser.add_argument('--images_dir', type=str)
    parser.add_argument('--faces_images_dir', type=str, default='faces_new'), 
    parser.add_argument('--faces_embeddings_pt', type=str, default='face_embedding.pt')

    parser.add_argument('--use_wandb', default=True, action='store_true')
    parser.add_argument('--wandb_project', type=str, default='FaceIt')
    parser.add_argument('--wandb_entity', type=str, default='bronershira')

    parser.add_argument('--device', type=str, default='mps')

    return parser.parse_args()


def init_wandb(args):
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args)
        
def extract_faces_from_images(use_wandb, images_data_dir, output_faces_images_dir, device):
    print(f'Extracting photos from {images_data_dir} to {output_faces_images_dir}')

    # Initialize MTCNN
    mtcnn = MTCNN(keep_all=True, device=torch.device(device), margin=2, thresholds=[0.8, 0.9, 0.9])
    stats_data = []

    for file_path in get_files_from_dir(os.path.join(images_data_dir, "*.*")):
        # Load an image
        image = Image.open(file_path)

        # Detect faces
        boxes, probs, landmarks = mtcnn.detect(image, True)
        i = 0
        file_name = os.path.basename(file_path).split(".")[0]
        i = 0
        
        if use_wandb:
            wandb.log({'faces_in_image': len(boxes) if boxes is not None else 0})
        if boxes is not None:
            for box, prob, landmark in zip(boxes, probs, landmarks):
                face_save_path = os.path.join(output_faces_images_dir, f'{file_name}_{i}.png'.format(i))
                x,y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
                if h * w < 10000 :
                    continue

                gray = cv2.cvtColor(np.array(image.crop(box)), cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                if variance <= 100:
                    continue
                if prob < 0.97 and variance < 350:
                    continue

                if use_wandb:
                    
                    wandb.log({'face_detected_prob': prob, 'laplacian_variance': variance})
                    print(f'{face_save_path} {prob}')
                
                stats_data.append({'face_file':face_save_path, 'prob': prob, 'laplace_variance': variance})       
                extract_face(image, box, save_path=face_save_path)
                i += 1

    pd.DataFrame(stats_data).to_csv('faces_stats')

def faces_to_embeddings(faces_data_dir, device, output_pt_path):
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    faces_data_dir = os.path.join(faces_data_dir, "*")
    faces_files = get_files_from_dir(faces_data_dir)
    print(f'Number of face images: {len(faces_files)}')

    with torch.no_grad():
        x = []
        y = []
        for face_file in tqdm.tqdm(faces_files, desc='Processing', total=len(faces_files)):
            y.append(face_file)
            x.append(resnet(transform(Image.open(face_file)).to(device).unsqueeze(0)).cpu())

    torch.save({'x': torch.concat(x), 'y':y}, output_pt_path)



if __name__ == "__main__":
    args = parse_args()
    print(args)
    init_wandb(args)
    device = args.device

    extract_faces_from_images(args.use_wandb, args.images_dir, args.faces_images_dir, device)
    faces_to_embeddings(args.faces_images_dir, device, args.faces_embeddings_pt)
    print('hi')