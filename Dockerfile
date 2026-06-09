# Step 1: Use an official lightweight Python base image.
# python:3.12-slim keeps the image size small and avoids unnecessary bloat.
FROM python:3.12-slim

# Step 2: Install system-level dependencies.
# We need build-essential for compiling C/C++ source (required by some libraries like chroma-hnswlib or numpy if built from source).
# We need curl for testing container health.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Set the working directory inside the container.
# All subsequent RUN, CMD, and COPY instructions will execute relative to this directory.
WORKDIR /app

# Step 4: Leverage Docker's layer caching for Python packages.
# By copying requirements.txt and running pip install BEFORE copying the rest of the application code,
# Docker caches this layer. Rebuilding after code changes will skip this step entirely!
COPY requirements.txt .

# Step 5: Install the Python dependencies.
# --no-cache-dir reduces image size by not storing the downloaded cache packages.
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of the application files into the container.
# This copies everything from the host workspace (where Dockerfile is) into the container's /app folder.
COPY . .

# Step 7: Document the ports that the container expects to listen on.
# - Port 7860: The Gradio web interface.
# - Port 6006: The Arize Phoenix tracing dashboard.
EXPOSE 7860 6006

# Step 8: Set the default execution command.
# When the container starts, it will run this command.
CMD ["python3", "src/app.py"]
