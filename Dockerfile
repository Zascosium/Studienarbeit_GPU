# Verwende ein Ubuntu-Image als Basis
FROM ubuntu:20.04

# Setze das Arbeitsverzeichnis
WORKDIR /app

# Installiere notwendige Systempakete
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libusb-1.0-0 \
    && apt-get clean

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Installiere TensorFlow ohne GPU-Unterstützung
RUN python3 -m pip install tensorflow-cpu==2.8.4

# Installiere die zusätzlichen Python-Abhängigkeiten
RUN python3 -m pip install pillow
RUN python3 -m pip install tflite-model-maker
RUN python3 -m pip install pycocotools
RUN python3 -m pip install numpy
RUN python3 -m pip install matplotlib
RUN python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral

# Kopiere die Dateien in das Container-Verzeichnis
COPY . /app

# Führe das Python-Skript zum trainieren aus
CMD ["python3", "combined_training.py"]

# Führe das Python-Skript zum trainieren aus
#CMD ["python3", "delete_unused_folders.py"]