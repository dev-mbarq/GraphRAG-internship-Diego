## graphsage_config.yaml

`graphsage_config.yaml` centralises the management of the hyper-parameters required to instantiate and train the GraphSAGE node-embedding model that powers this project. It is comprised of the following fields:

-   `model_params → channels`: An ordered list that simultaneously fixes the width of each GraphSAGE layer and the number of neighbourhood “hops” aggregated. The first value is the dimensionality of the pre-computed semantic embedding stored on every node; each subsequent value becomes the hidden size of a new layer. Consequently, the list length L determines the receptive-field depth (L – 1 hops) and the last value is the dimensionality of the final node embedding produced by the network.

- `loader_params → num_neighbors, batch_size, shuffle`: Controls the PyTorch-Geometric NeighborLoader. `num_neighbors` must contain L – 1 integers indicating how many neighbours to sample at hop 1, hop 2, …; `batch_size` sets how many root nodes are processed per optimisation step; `shuffle` decides whether those root nodes are reshuffled at the start of each epoch.

- `input_data → graph_file_name`: Relative path from the root of the repository to a .pkl file that holds the graph to be processed (Networkx). Every node must already carry a pre-assigned text embedding under the attribute name 'embedding'.

- `optimizer_params → learning_rate`: The learning rate passed to the Adam optimiser.

- `training_params → num_epochs`: Number of epochs to train.

- `bundle_tag`: Prefix used to name the directory where the outputs of the training process under `data/retrieval_bundles/`, including model weights, the network's hyperparameters, the input graph with the newly produced embeddings by the network and other relevant data and reports regarding the evolution of the training process.

Example: with `channels`: [1024, 768, 512] and `num_neighbors`: [15, 15] the network ingests 1024-d semantic vectors, applies two GraphSAGE layers that aggregate 15 first-hop and 15 second-hop neighbours respectively, and returns a 512-d embedding for each node. These values can be tuned to balance representation power, receptive-field size, training speed and memory footprint.