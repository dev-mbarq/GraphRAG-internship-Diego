# GraphRAG-internship-Diego

The goal of this project is to explore how vector-embedded knowledge graphs can serve as knowledge bases for Retrieval-Augmented Generation (RAG) systems. Concretely, graphs whose nodes are mapped to high-dimensional vectors, which can potentially enable the use of hybrid retrieval strategies that marry the efficiency and flexibility of vector search with the structured, higher-factuality results of retrieval based on knowledge graphs and graph search. This repository concentrates on methods for generating node embeddings for document graphs that jointly capture the semantic content of text-attributed nodes and the topological signals propagated from their neighbourhood via Graph Neural Networks (GNNs). Although the repository contains some experimental exercises on retrieval pipelines, the core emphasis here is on GNN-based methods for node-embedding generation.

### Repository structure

- Directory __config/__ : Contains `config.yaml`, where all parameters for the node-embedding model and its training pipeline are defined. For further details on the components and structure of __config.yaml__, check the README.md provided under `config/`.

- Directory __data/__ : This directory is meant to store all the data of the project, including inputs, intermediate artifacts and final outputs. All scripts and notebooks reference this directory when loading inputs or saving results.

- Directory __notebooks/__ : Jupyter-style notebooks organized into three main subfolders (each targeting a different dataset and covering data exploration, graph construction and semantic-embedding workflows), plus three cross-dataset analyses:
     - __graphsage_embeddings.ipynb__: training GraphSAGE for node embeddings.
     - __parallel_retrieval_eval.ipynb__: assessing retrieval performance.
     - __embedding_space_exploration.ipynb__: visual analysis of embedding spaces.

- Directory __prompts/__ : Stand-alone LLM prompts defined are used through the execution of different code segments of the project.

- Directory __src/__ : Core Python modules split into two packages:
    - __riziv/__ – preprocessing routines specific to RIZIV data
    - __main/__ – model definitions, training logic and related components.
    The key entry point is `src/main/graphsage_embeddings.py`, which instantiates a GNNs and orchestrates its training for the production of node embeddings given the inputs. For further details on the functioning of `src/main/graphsage_embeddings.py`, check the README.md provided under `src/main/`.

Regarding the project's environment and execution the following is also provided:

- __requirements.txt__: Lists all Python dependencies required to run the codebase.

- __dockerfile__: Builds a Docker image pre-configured with the project’s environment.

- __docker-compose.yml__: Defines the service to build the Docker image and mage and start the container that hosts the full project runtime in a single command.

Important: This project targets Python 3.11. The Dockerfile already specifies it, but if you install dependencies manually, please ensure you’re using Python 3.11 to avoid compatibility issues.

### Interact with the project through the provided Docker resources

The repository ships with a ready-to-use Docker setup that builds the entire Python 3.11 environment (PyTorch + PyG + all dependencies), launches a Jupyter Notebook server and mounts your local source tree inside the container so that any change on disk is instantly visible in Jupyter.

#### 1. Build the image and start the stack (Jupyter included)

Executing `docker compose up --build` for the first time creates the image from the dockerfile, an operation that may take a few minutes to complete. Subsequent runs reuse the already created image starting it almost instantly.

Additionally, Jupyter Notebook is started automatically and is reachable at http://localhost:8888 without any token or password required. All project files are available inside the container under `/GraphRAG-internship-Diego` mirroring the repository outside the container through a docker volume.

#### 2. Open an interactive shell in the running container

With the container running, it is possible to run `docker compose exec graphrag bash` to open an interactive shell inside the container that can be used to run scripts or perform other operations while the Notebook server continues to run in the background.

__CUDA discalaimer__
