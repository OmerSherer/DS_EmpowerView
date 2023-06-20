# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Add the current directory contents into the container at /app
ADD . /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Make port 80 available to the world outside this container
EXPOSE 3000

# Run app.py when the container launches
CMD ["gunicorn", "-b", ":3000", "--workers", "1", "app:app"]
