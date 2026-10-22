# JetPack 6 (L4T R36) base image. For JetPack 5 use latest-jetson-jetpack5.
ARG BASE_IMAGE=ultralytics/ultralytics:latest-jetson-jetpack6
FROM ${BASE_IMAGE}
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["python3", "backend/check_env.py"]
