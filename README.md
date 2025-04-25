# GraphRAG-internship-

Goal: Explore node embedding generation

### Repository structure

- Directory __config/__ : Contains __config.yaml__, where all parameters for the node-embedding model and its training pipeline are defined.

- Directory __data/__ : This directory is meant to store all the data of the project, including inputs, intermediate artifacts and final outputs. All scripts and notebooks reference this directory when loading inputs or saving results.

- Directory __notebooks/__ : Jupyter-style notebooks organized into three main subfolders (each targeting a different dataset and covering data exploration, graph construction and semantic-embedding workflows), plus three cross-dataset analyses:
     - __graphsage_embeddings.ipynb__: training GraphSAGE for node embeddings.
     - __parallel_retrieval_eval.ipynb__: assessing retrieval performance.
     - __embedding_space_exploration.ipynb__: visual analysis of embedding spaces.

- Directory __prompts/__ : Stand-alone LLM prompts defined are used through the execution of different code segments of the project.

- Directory __src/__ : Core Python modules split into two packages:
    - __riziv/__ – preprocessing routines specific to RIZIV data
    - __main/__ – model definitions, training logic and related components.
    The key entry point is src/main/graphsage_embeddings.py, which instantiates the GraphSAGE model and orchestrates its training given the inputs.

Regarding the project's environment and execution the following is also provided:

- __requirements.txt__: Lists all Python dependencies required to run the codebase.

- __dockerfile__: Builds a Docker image pre-configured with the project’s environment.

Important: This project targets Python 3.11. The Dockerfile already specifies it, but if you install dependencies manually, please ensure you’re using Python 3.11 to avoid compatibility issues.
