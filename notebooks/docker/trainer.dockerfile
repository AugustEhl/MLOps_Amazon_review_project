# Base image
FROM python:3.9
# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*
#copy files
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
# setting workdir
WORKDIR /
#RUN python3 -m pip install --upgrade pip setuptools wheel                                                                                                                                                                                                
#RUN python3 -m pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]