print("Starting GraphSAGE embeddings training...")

########################################################
# A. Import relevant dependencies
########################################################

print("Importing dependencies...")

import os
import pickle
import sys

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected
from torch_geometric.utils.convert import from_networkx
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR


# This will work in scripts where __file__ is defined
current_dir = os.path.dirname(os.path.abspath(__file__))
# Project root: two levels above src/main/
project_root = os.path.abspath(os.path.join(current_dir, "../.."))

src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from graph_formatting_utils import prepare_graph_for_gnn
from loss_functions import unsupervised_loss_V0, unsupervised_loss_V1
from node_embedding_models import GraphSAGE, GraphSAGE_V2
from training_utils import train_in_cpu, train_in_gpu, train_model_in_gpu_V2

print("Importing dependencies... Done \n")

########################################################
# B. Check CUDA status
########################################################

print("CUDA Available:", torch.cuda.is_available())
print(
    "GPU Name:",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected",
)
print("CUDA Device Count:", torch.cuda.device_count(), "\n")

########################################################
# C. Load config data
########################################################

print("Loading config data...")

# Load GraphSAGE config file
config_file_path = os.path.join(project_root, "config", "graphsage_config.yaml")

with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

input_graph_file_name = config["input_data"]["graph_file_name"]
input_graph_embedding_dim = config["input_data"]["embedding_dim"]

graphsage_channels = config["model_params"]["channels"]

loader_num_neighbors = config["loader_params"]["num_neighbors"]
loader_batch_size = config["loader_params"]["batch_size"]
loader_shuffle = config["loader_params"]["shuffle"]

training_num_epochs = config["training_params"]["num_epochs"]
optimizer_learning_rate = config["optimizer_params"]["learning_rate"]

bundle_tag = config["bundle_tag"]

print("Loading config data... Done \n")

########################################################
# D. Load data and train embedding model
########################################################

print("Loading input graph...")

# Load and format graph
graph_path = os.path.join(project_root, "data", input_graph_file_name)
with open(graph_path, "rb") as f:
    G = pickle.load(f)

formatted_G, incidences = prepare_graph_for_gnn(
    G, embedding_dim=input_graph_embedding_dim
)

# Convert the NetworkX graph to a PyTorch Geometric Data object
data = from_networkx(formatted_G)

# Ensure the graph is undirected
data.edge_index = to_undirected(data.edge_index)

# Create data attribute "x" containing the embeddings of each node complying with the PyTorch Geometric API
data.x = data.embedding
del data.embedding

print("Loading input graph... Done \n")

print("Setting up model and training pipeline...")

# Instantiate the GraphSAGE model
model = GraphSAGE_V2(channels=graphsage_channels)

# Set device for model training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

# Instantiate the NeighborLoader for mini-batch training
train_loader = NeighborLoader(
    data,
    num_neighbors=loader_num_neighbors,  #  neighbors for the first layer, 15 for the second
    batch_size=loader_batch_size,  # Batch size
    shuffle=loader_shuffle,
)

# Define optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=optimizer_learning_rate
)  # (old optimizer)

# Define scaler if GPU is available
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

print("Setting up model and training pipeline... Done \n")

print("Training model...")

if not torch.cuda.is_available():
    training_outputs = train_in_cpu(
        model,
        train_loader,
        optimizer,
        num_epochs=training_num_epochs,
        loss_fn=unsupervised_loss_V1,
        debug=False,
        plot_eval=True,
    )

elif torch.cuda.is_available():
    # V1
    training_outputs = train_in_gpu(
        model,
        train_loader,
        optimizer,
        num_epochs=training_num_epochs,
        loss_fn=unsupervised_loss_V1,
        debug=False,
        plot_eval=True,
    )

print("Training model... Done \n")

########################################################
# E. Save embeddings, model weights and config
########################################################

print("Saving embeddings, model weights and config...")

# Add Graph_SAGE embeddings to the baseline graph

## -> Move data to the same device as the model
device = next(model.parameters()).device
data_x = data.x.to(device)
data_edge_index = data.edge_index.to(device)

## -> Obtain final embeddings from the trained model
with torch.no_grad():
    final_emb = model(data_x, data_edge_index)  # shape [num_nodes, embedding_dim]
    final_emb_np = final_emb.cpu().numpy()

## -> Add them back to the cleaned_G graph
list_of_nodes = list(G.nodes())  # Must match the node ordering in data
for i, node in enumerate(list_of_nodes):
    # Store as a NumPy array (or you could store as a list if you prefer)
    G.nodes[node]["hybrid_embedding"] = final_emb_np[i]

# Define tags for the output files
training_n_hops = len(graphsage_channels) - 1
channels_str = "-".join([str(i) for i in graphsage_channels])
training_num_epochs

# Check if retrieval_bundles directory exists
retrieval_bundles_dir = os.path.join(project_root, "data", "retrieval_bundles")
os.makedirs(retrieval_bundles_dir, exist_ok=True)

# Create bundle directory
bundle_directory = os.path.join(
    retrieval_bundles_dir,
    f"{bundle_tag}_{training_n_hops}hop_{training_num_epochs}epochs_{channels_str}",
)

os.makedirs(bundle_directory, exist_ok=True)

# Save the processed graph
output_graph_bundle_path = os.path.join(bundle_directory, "graph.pkl")
with open(output_graph_bundle_path, "wb") as f:
    pickle.dump(G, f)

# Save the trained model
output_model_bundle_path = os.path.join(bundle_directory, "graphsage.pth")
torch.save(model.state_dict(), output_model_bundle_path)

# Save config dictionary
config_bundle_path = os.path.join(bundle_directory, "config.yaml")
with open(config_bundle_path, "w") as f:
    yaml.dump(config, f)

# Extract figures from dictionary
loss_fig = training_outputs.pop("loss_fig")
norm_fig = training_outputs.pop("norm_fig")

# Save figures
loss_fig.savefig(os.path.join(bundle_directory, "loss_evolution.jpg"))
norm_fig.savefig(os.path.join(bundle_directory, "norm_evolution.jpg"))
plt.close("all")

# Save metrics dictionary (now without figures)
metrics_path = os.path.join(bundle_directory, "training_metrics.pkl")
with open(metrics_path, "wb") as f:
    pickle.dump(training_outputs, f)

print("Saving embeddings, model weights and config... Done \n")

print("Training completed successfully!")
