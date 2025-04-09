FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Set the working directory
WORKDIR /crowd-follow-nav

COPY ./PySocialForce /crowd-follow-nav/PySocialForce
COPY ./Python-RVO2 /crowd-follow-nav/Python-RVO2

RUN pip install -r /crowd-follow-nav/requirements.txt

RUN apt-get update && apt-get install -y cmake
RUN apt-get update && apt-get install -y cmake build-essential git

# Install Python-RVO2
RUN cd /crowd-follow-nav/Python-RVO2 && \
    pip install Cython && \
    python setup.py build && \
    python setup.py install

# Install PySocialForce in editable mode
RUN cd /crowd-follow-nav/PySocialForce && \
    pip install -e '.[test,plot]'
    
    

CMD ["/bin/bash"]