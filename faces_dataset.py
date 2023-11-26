import os
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import cv2


class EmbeddingFacesDataSet(Dataset):
    def __init__(self):
        self.embedding_dict = torch.load('face_embedding.pt')
        self.x = self.embedding_dict['x']
        self.y = self.embedding_dict['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def is_blurry(image_path, threshold=100):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    def get_all(self):
        data_loader = DataLoader(self, batch_size=len(self), shuffle=True, num_workers=4)
        for _, (images, labels) in enumerate(data_loader):
            return (images, labels)
    
    