FROM python:3.12

# Install OS dependencies needed for build
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     g++ \
#     python3-dev \
#     git \
#     && rm -rf /var/lib/apt/lists/*
# # Clone and install GED4py
# RUN pip install numpy cython

# RUN git clone https://github.com/chilligerchief/GED4py.git /ged4py
# WORKDIR /ged4py
# RUN pip install .

WORKDIR /app

# Install required Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your code
COPY . .

ENV LD_LIBRARY_PATH=/app/yadt:$LD_LIBRARY_PATH

# Make dTcmd executable
RUN chmod +x yadt/dTcmd
#RUN ./yadt/dTcmd -fd ./datasets/compas-scores-two-years.data -fm ./datasets/compas-scores-two-years.names -sep ';' -d ./datasets/compas-scores-two-years.dot

# Your command here
CMD ["python", "evaluate_lore_graph.py"]