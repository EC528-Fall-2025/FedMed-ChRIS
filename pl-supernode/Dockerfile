  1 FROM python:3.10-slim
  2 WORKDIR /usr/local/src/app
  3
  4 # Install Python deps
  5 COPY requirements.txt .
  6 RUN pip install --no-cache-dir -r requirements.txt
  7
  8 # Copy only what you need at runtime
  9 COPY app.py .
 10
 11 # Run the ChRIS plugin entrypoint (app.py)
 12 ENTRYPOINT ["python3", "app.py"]
 13
