# Stage 1: Build the React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
# Copy frontend package files
COPY frontend/package*.json ./
RUN npm install
# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# Stage 2: Serve with FastAPI and Python
FROM python:3.12-slim
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files and models
COPY api.py .
COPY models/ ./models/

# Copy the built React app from the builder stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI application on port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
