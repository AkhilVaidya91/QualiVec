# Dockerfile for QualiVec Streamlit Demo

# 1. Base Image
FROM python:3.12-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Install uv - the fast Python package manager
RUN pip install --no-cache-dir uv

# 5. Copy dependency definition files and README (required for package build)
COPY pyproject.toml uv.lock README.md ./

# 6. Copy source code (needed for package installation)
COPY src/ ./src/

# 7. Install Python dependencies using uv
# 'uv pip install .' reads pyproject.toml and installs the project dependencies
RUN uv pip install --system --no-cache-dir .

# 8. Copy the rest of the application source code
# Make sure you have a .dockerignore file to exclude .venv
COPY . .

# 9. Expose the port Streamlit runs on
EXPOSE 8501

# 10. Add a health check to verify the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 11. Define the entry point to use the run_demo.py script via uv
# ENTRYPOINT ["uv", "run", "app/run_demo.py"]
ENTRYPOINT ["python", "app/run_demo.py"]