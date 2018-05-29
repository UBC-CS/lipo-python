FROM gcc:5

RUN apt-get update && \
    apt-get install libboost-dev cmake cmake-curses-gui g++ -y && \
    apt-get install python-dev -y && \
    apt-get install python-numpy -y 
    
RUN git clone https://github.com/rmcantin/bayesopt && \
    cd bayesopt && \
    cmake -DBAYESOPT_PYTHON_INTERFACE=ON . && \ 
    make  && \ 
    make install
