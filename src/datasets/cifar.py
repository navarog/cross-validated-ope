import os
import pickle

import torch
import numpy as np
import torchvision.transforms as T
from torchvision.datasets import CIFAR100


def get_dataset(embedding_model = "cifar100_resnet56", data_dir="data/datasets", repo_id = "chenyaofo/pytorch-cifar-models", batch_size=1000):
    cached_data = f"{data_dir}/{embedding_model}.pkl"
    if os.path.exists(cached_data):
        with open(cached_data, 'rb') as f:
            X, y = pickle.load(f)
        return X, y
    
    def transform(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]):
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(repo_id, embedding_model, pretrained=True).to(device)
    # remove the classification head 
    head = model.fc
    model.fc = torch.nn.Identity().to(device)
    
    train = CIFAR100(data_dir, download=not os.path.exists(f"{data_dir}/cifar-100-python/train"), train=True, transform=transform())
    test = CIFAR100(data_dir, download=not os.path.exists(f"{data_dir}/cifar-100-python/test"), train=False, transform=transform())
    merged = torch.utils.data.ConcatDataset([train, test])
    loader = torch.utils.data.DataLoader(merged, batch_size=batch_size, shuffle=False)
    
    context_dim = head.in_features
    n_actions = head.out_features
    X = np.zeros((len(merged), context_dim))
    y = np.zeros((len(merged), n_actions))
    for i, (inputs, targets) in enumerate(loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs)
            targets_hat = head(outputs) - 3 # lower the overal reward
        
        X[i*batch_size:(i+1)*batch_size] = outputs.cpu().numpy()
        y[i*batch_size:(i+1)*batch_size] = torch.sigmoid(targets_hat).cpu().numpy()
    
    with open(cached_data, 'wb') as f:
        pickle.dump((X, y), f)
        
    return X, y