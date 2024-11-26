# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the CSV file
COPY house_prices.csv .

# Copy the rest of the application code
COPY . .

EXPOSE 8000 8501
# Command to run the application
CMD ["sh", "-c", "uvicorn predictor:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
