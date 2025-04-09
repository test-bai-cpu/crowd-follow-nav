FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Set the working directory
WORKDIR /workdir

COPY ./PySocialForce /workdir/PySocialForce
COPY ./Python-RVO2 /workdir/Python-RVO2


COPY ./requirements.txt /workdir/requirements.txt

RUN pip install -r /workdir/requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    cmake \
    build-essential \
    git

# Install Python-RVO2
RUN cd /workdir/Python-RVO2 && \
    pip install Cython && \
    python setup.py build && \
    python setup.py install

# Install PySocialForce in editable mode
RUN cd /workdir/PySocialForce && \
    pip install -e '.[test,plot]'
    
    

CMD ["/bin/bash"]