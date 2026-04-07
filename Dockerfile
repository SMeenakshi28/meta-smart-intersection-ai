# Use Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenGL (needed for HighwayEnv rendering)
RUN apt-get update && apt-get install -y \
    freeglut3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy all files to the container
COPY . .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit uses on Hugging Face
EXPOSE 7860
EXPOSE 8000

# Run the app on the specific port Hugging Face requires
CMD ["python", "app.py"]
