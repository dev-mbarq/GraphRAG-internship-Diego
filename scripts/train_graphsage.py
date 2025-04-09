import os
import sys
import pickle
import random

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected
from torch_geometric.utils.convert import from_networkx

# Add "src" path to Python path
sys.path.append(os.path.abspath("../src"))

# Import custom graph formatting function
from graph_formatting_utils import format_graph_for_graphsage
from models import GraphSAGE
from losses import unsupervised_loss
from train import train

# Check CUDA status
print("CUDA Available:", torch.cuda.is_available())
print(
    "GPU Name:",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected",
)
print("CUDA Device Count:", torch.cuda.device_count())

# Load baseline graph
with open("../data/multihop_graph_w_sem_embeddings.pkl", "rb") as f:
    G = pickle.load(f)

cleaned_G = format_graph_for_graphsage(G, embedding_dim=1024)

# Convert the NetworkX graph to a PyTorch Geometric Data object
data = from_networkx(cleaned_G)

# Ensure the graph is undirected
data.edge_index = to_undirected(data.edge_index)

# Create data attribute "x" containing the embeddings of each node complying with the PyTorch Geometric API
data.x = data.embedding

# Instantiate the GraphSAGE model
model = GraphSAGE(
    in_channels=1024,  # Input features (BGE-M3 embeddings)
    hidden_channels=512,  # First hidden layer (alto para máxima capacidad)
    out_channels=256,  # Output embeddings (más ricos)
    num_layers=2,  # Mantenemos 2 capas (2 hops)
)

## Instantiate the NeighborLoader for mini-batch training
train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 15],  # 25 neighbors for the first layer, 15 for the second
    batch_size=512,  # Batch size
    shuffle=True,
)

# Set device for model training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

# Check model device
print("Model in:", next(model.parameters()).device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define scaler
scaler = torch.cuda.amp.GradScaler()

train(
    model,
    train_loader,
    optimizer,
    device="cpu",
    num_epochs=5,
    loss_fn=unsupervised_loss,
    scaler=None,
)


# scaler = torch.cuda.amp.GradScaler()
# train(model, train_loader, optimizer, device='cuda', num_epochs=100, loss_fn=loss_fn, scaler=scaler)
