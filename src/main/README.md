## `graphsage_embeddings.py`  —  train a GraphSAGE model end-to-end

`graphsage_embeddings.py` is the main entry-point for producing hybrid node embeddings that combine the semantics of each node (pre-computed text embedding) with the topological signal propagated through a GraphSAGE network.

At a high level the script:

1. Reads a YAML config that specifies model width, sampling strategy, optimiser and training schedule;

2. Loads a pickled NetworkX graph whose nodes already carry a semantic embedding;

3. Converts that graph into a PyTorch-Geometric Data object, builds a multi-layer GraphSAGE model, and trains it with an unsupervised contrastive loss on CPU or GPU;

4. Writes a complete “bundle” to disk so the experiment can be reproduced or plugged into downstream retrieval pipelines.

### Required inputs

1. `config/graphsage_config.yaml`: Config file containing model and training hyperparameters (See README.md in `config/`).

2. Graph pickle (.pkl): A NetworkX graph in which every node contains an attribute called "embedding" that holds a a NumPy vector representing semantic information. 

### Produced outputs

After training, the script creates a directory inside `data/retrieval_bundles/` whose name encodes the provided bundle tag, hop depth, epoch count and channel sizes, for example: `bsard_V2_2hop_10epochs_1024-768-512/`.

The folder contains:

- `graph.pkl` – the original graph plus a new node attribute "hybrid_embedding" with the learned GraphSAGE vector.

- `graphsage.pth` – the model weights for re-loading.

- `config.yaml` – an exact copy of the configuration (`graphsage_config.yaml`) used in a particular run.

- `loss_evolution.jpg` and `norm_evolution.jpg` – plots that track and represent loss and embedding-norm statistics over epochs and batches.

- `training_metrics.pkl` – a pickled dictionary with the raw metric history (loss per batch, timing, etc.).

### Running the script

Run the command `python src/main/graphsage_embeddings.py` inside the provided Docker container (or any Python 3.11 environment satisfying requirements.txt). The script locates the YAML config automatically; ensure that graph_file_name points to the correct graph before launching.