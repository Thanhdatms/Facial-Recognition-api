FROM arm64v8/python:3.11
# Set non-interactive mode to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    libopencv-dev \
    libjpeg-dev \
    libtiff-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create a working directory
WORKDIR /app

# Copy application files
COPY . .

# Create a virtual environment
RUN python3 -m venv /app/venv

# Activate virtual environment and install dependencies
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Install CPU-only versions of torch and torchvision
RUN /app/venv/bin/pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Set the virtual environment as default Python
ENV PATH="/app/venv/bin:$PATH"

# Expose Flask API port (if needed)
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
