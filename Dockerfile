# # Use the official Python 3.10 image as the base image
FROM python:3.11

# Set the working directory
WORKDIR /ollama-python

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy the project files into the working directory
COPY . .

# Install project dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --only main
     
# Install pre-commit and set up the hooks
RUN poetry run pre-commit install
# Specify the command to run your application
CMD ["sh", "-c", "while :; do sleep 10; done"]

