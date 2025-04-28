- Inputs: Config.yml, Graph with semantic embeddings

- Outputs: (Saved in data/retreival_bundles/ in ad hoc folder) Model weights (.pth), Processed graph with a new attribute "SAGE_embedding", Config dictionary employed for training the model for reproducibility, plot of the evolution of the loss and embedding norms per epoch and batch. File containing the trainign data.