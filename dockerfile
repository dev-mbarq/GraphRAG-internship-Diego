# Use a lightweight Python image
FROM python:3.11.6-slim

# Install Git and other system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /GaphRAG-internship-Diego

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Default command 
# CMD ["python", XXX.py] 