# Start from the official python base image
FROM python:3.9

# Set Current Working Directory to /code for keeping requirements.txt file.
WORKDIR /code

# copy file with the requirements to the /code directory.
COPY ./requirements.txt /code/requirements.txt

# Install the package dependencies in the requirements file.
# The --no-cache-dir option tells pip to not save the downloaded packages locally, 
# as that is only if pip was going to be run again to install the same packages, 
# but that's not the case when working with containers.
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# copy the ./app directory inside the /code directory
COPY ./api /code/api

# Set the command to run the uvicorn server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]
