FROM python:3.8

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
CMD ["python", "ba_lore.py"]