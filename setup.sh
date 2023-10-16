#!/bin/sh

CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir 

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 

pip install -r requirements_GGML.txt 