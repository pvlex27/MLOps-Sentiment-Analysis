# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
# Install dependencies without '--no-cache-dir'
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose port 8080 for FastAPI
EXPOSE 8080


#without relosd for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]