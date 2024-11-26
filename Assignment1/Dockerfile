# Use a smaller base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the required files
COPY titanic_eda.py .
COPY main.py .
COPY titanic.csv .

# Command to run the application
CMD ["python", "main.py"]
