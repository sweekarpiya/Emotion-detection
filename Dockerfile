FROM python:3.7

WORKDIR /env/
# Set up virtual environment

RUN python3 -m venv /env/emotion
RUN /bin/bash -c "source /env/emotion/bin/activate"

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /api
ADD . /api

WORKDIR /ml
ADD . /ml

# Run the application:
CMD ["python3", "ml/src/main.py"]



