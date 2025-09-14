# Use an official CUDA runtime as a parent image
FROM nvcr.io/nvidia/pytorch:24.12-py3
#nvidia/cuda:12.6.1-devel-ubuntu22.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y python3-pip git && ldconfig

# Install torch, torchvision, torchao
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 torchao
#RUN pip install torchtune
# Install torchtune and its dependencies, including dev dependencies
RUN pip install --upgrade pip setuptools
RUN pip install -e .

# Add /usr/local/bin to PATH
ENV PATH="/usr/local/bin:${PATH}"
