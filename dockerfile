# Use a lightweight Python image
FROM python:3.11.6-slim

# Install Git and other system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /GaphRAG-internship-

# Install torch
RUN pip install --no-cache-dir torch==2.1.0 --extra-index-url https://download.pytorch.org/whl/cpu

# PyG
RUN pip install --no-deps \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir notebook

# Copy the entire project code
COPY . .

EXPOSE 8888

# Default command 
# CMD ["python", XXX.py] 