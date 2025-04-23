# Use an official Miniconda image as the base image
FROM continuumio/miniconda3:4.12.0

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false

# Set the working directory in the container
WORKDIR /app

# Copy the cleaned conda environment file
COPY docker_conda_env.yml /tmp/conda_env.yml

# Create the Conda environment from the environment.yml file
RUN conda env create -f /tmp/conda_env.yml

# Make sure conda is initialized for bash
RUN echo "conda activate conda_qres_marl_docker" >> ~/.bashrc

# Set default shell to use the conda environment
SHELL ["conda", "run", "--no-capture-output", "-n", "conda_qres_env", "/bin/bash", "-c"]

# Copy the application code
COPY . /app

# Install additional pip dependencies if needed (optional)
# RUN pip install some-package

# Set the entry point to run the main script
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "conda_qres_env", "python", "imprl-infinite-horizon/examples/train_and_log_parallel.py"]

# Allow for command-line arguments (e.g., WANDB_API_KEY)
CMD ["--WANDB_API_KEY", "default_value"]
