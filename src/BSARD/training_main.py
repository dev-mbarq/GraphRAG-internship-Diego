import os
import pickle
import sys

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import importlib
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected
from torch_geometric.utils.convert import from_networkx
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

try:
    # This will work in scripts where __file__ is defined
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming "src" is parallel to the script folder
    project_root = os.path.abspath(os.path.join(current_dir, "..",".."))
except NameError:
    # In notebooks __file__ is not defined: assume we're in notebooks/
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from main.graph_formatting_utils import prepare_graph_for_gnn
from main.loss_functions import unsupervised_loss_V0, unsupervised_loss_V1
from main.node_embedding_models import *
from main.training_utils import train_in_cpu, train_in_gpu, train_in_gpu_with_checkpoints

def main_training_pipeline(config):

    # Pop config parameters

    # Model
    model_name = config['model']['model_name']
    graphsage_channels = config['model']['model_channels']
    input_graph_embedding_dim = config['model']['model_channels'][0] 

    # Input data
    input_graph_file_name = config['input_data']['graph_file_name']

    # Loader
    loader_num_neighbors = config['loader_params']['num_neighbors']
    loader_batch_size = config['loader_params']['batch_size']
    loader_shuffle = config['loader_params']['shuffle']

    # Training
    training_num_epochs = config['training_params']['num_epochs']

    # Optimizer
    optimizer_learning_rate = config['optimizer_params']['learning_rate']

    # Bundle
    bundle_tag = config["bundle_tag"]

    def load_model(model_name, channels, **kwargs):
        # 1) Importa dinámicamente el módulo donde están definidas las clases
        m = importlib.import_module("main.node_embedding_models")
        # 2) Saca de ese módulo la clase con el mismo nombre que MODEL_NAME
        try:
            ModelClass = getattr(m, model_name)
        except AttributeError:
            raise ValueError(f"Model {model_name!r} not found in models.py")
        # 3) Instáncialo pasándole lo que necesites
        return ModelClass(channels, **kwargs)
    
    # Load model class
    model = load_model(model_name, graphsage_channels)

    # Load and format graph
    graph_path = os.path.join(project_root, "data", input_graph_file_name)
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    formatted_G, incidences = prepare_graph_for_gnn(G, embedding_dim=input_graph_embedding_dim)

    # Convert the NetworkX graph to a PyTorch Geometric Data object
    data = from_networkx(formatted_G)

    # Ensure the graph is undirected
    data.edge_index = to_undirected(data.edge_index)

    # Create data attribute "x" containing the embeddings of each node complying with the PyTorch Geometric API
    data.x = data.embedding
    del data.embedding

    # Set device for model training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    model = model.to(device)

    # Instantiate the NeighborLoader for mini-batch training
    train_loader = NeighborLoader(
        data,
        num_neighbors=loader_num_neighbors,  #  neighbors for the first layer, 15 for the second
        batch_size=loader_batch_size,  # Batch size
        shuffle=loader_shuffle
    )

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_learning_rate) # (old optimizer)

    # Define scaler if GPU is available
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    if not torch.cuda.is_available():
        training_outputs = train_in_cpu(model, train_loader, optimizer, num_epochs=training_num_epochs, loss_fn=unsupervised_loss_V1, debug=False, plot_eval=True)

    elif torch.cuda.is_available():
        #training_outputs = train_in_gpu(model, train_loader, optimizer, num_epochs=training_num_epochs, loss_fn=unsupervised_loss_V1, debug=False, plot_eval=True)
        training_outputs = train_in_gpu_with_checkpoints(model, train_loader, optimizer, num_epochs=training_num_epochs, loss_fn=unsupervised_loss_V1, debug=False, plot_eval=True, 
                                                         checkpoint_interval=5, checkpoint_dir=os.path.join(project_root, "data", "BSARD_dataset", "checkpoints"))

    #  Add Graph_SAGE embeddings to the baseline graph

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
    training_n_hops = len(graphsage_channels)-1
    channels_str = "-".join([str(i) for i in graphsage_channels])
    training_num_epochs

    # Check if retrieval_bundles directory exists
    retrieval_bundles_dir = os.path.join(project_root, "data", "retrieval_bundles")
    os.makedirs(retrieval_bundles_dir, exist_ok=True)

    # Create bundle directory
    bundle_directory = os.path.join(
        retrieval_bundles_dir, f"{bundle_tag}_{training_n_hops}hop_{training_num_epochs}epochs_{channels_str}"
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

    # Save metrics dictionary (now without figures)
    metrics_path = os.path.join(bundle_directory, "training_metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(training_outputs, f)

    return training_outputs, bundle_directory