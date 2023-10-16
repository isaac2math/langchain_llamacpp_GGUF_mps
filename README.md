<h3 align="center"> LLM finetuning and deployment using CUDA </h3>
<h4 align="center"> Isaac W. N. Xu </h4>

### File structure

```txt
├── data
│   ├── feature
│   │   ├── os-1.txt
│   │   ├── os-2.txt
│   │   └── os-3.txt
│   ├── raw
│   │   └── open-source.pdf
│   ├── READEME.md
│   └── transform
├── langchain_llama2.yaml
├── model
│   ├── llama-2-13b.ggmlv3.q3_K_L.bin
│   ├── llama-2-13b.ggmlv3.q4_K_M.bin
│   ├── llama-2-13b.ggmlv3.q6_K.bin
│   ├── llama-2-13b-GPTQ
│   ├── llama-2-70b.ggmlv3.q3_K_L.bin
│   ├── llama-2-70b.ggmlv3.q4_K_M.bin
│   ├── llama-2-70b.ggmlv3.q5_K_M.bin
│   ├── llama-2-7b.ggmlv3.q3_K_L.bin
│   ├── llama-2-7b.ggmlv3.q4_K_M.bin
│   ├── llama-2-7b.ggmlv3.q6_K.bin
│   ├── Llama-2-7B-GPTQ
│   ├── model_download.sh
│   └── README.md
├── notebook
│   └── trial.ipynb
├── params.yaml
├── README.md
├── requirements.txt
└── src
    ├── embed.py
    ├── inference.py
    ├── inf_llama2-13B-q3.py
    ├── inf_llama2-13B-q4.py
    ├── inf_llama2-13B-q6.py
    ├── inf_llama2-7B-q3.py
    ├── inf_llama2-7B-q6.py.py
    ├── test3.py
    └── test.py
```

### Deployment: assume you are using Unix

- Install Miniconda:
 
```sh
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash miniconda.sh
```

- Clone the repo:
 
```sh
git clone https://github.com/isaac2math/langchain_llama2.git
```

- Set up conda environment

```sh
cd langchain_llama2

conda create -n langchain_llama2 python=3.10 -y
conda activate langchain_llama2

# if deploying GPTQ model, make sure the pytorch wheel and the cuda of this env share the same version
# otherwise cuda error would appear

sudo apt-get install -y cmake pkg-config

conda env config vars set LLAMA_CUBLAS=1

conda activate langchain_llama2

```

for NV-GPU:

```sh
conda install cuda -c nvidia/label/cuda-11.8.0

conda env config vars set LLAMA_CUBLAS=1

conda activate langchain_llama2

CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir 

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-cache-dir 

pip install -r requirements_GGML.txt

```

- Download pretrained-LLM

```sh
mkdir -p data/feature data/raw data/transform models
bash model/model_download.sh
```

### Data Preprocessing

- tba

### Training

- tba

### Fine-tuning

- tba

### Pruning

- tba

### Quantization

- tba

### Inference Deployment

- Create a vectorstore using word embedding from the documents in `data/feature`
    ```sh
    python src/embed.py
    ```

- Make inferences and give instructions based on the vectorstore in `data/vectorstore`
    ```python
    python src/inference.py
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


#### Usage

- This repo is a PoC for in-house and offline LLM application
- Currently this repo relies on 
  - `langchain` : 
  - `llama-cpp` : 
  - `chroma` : 
  - `sbert` : 
  - `gpt4all` :


#### Roadmap

- [ ] Replace Python with Cuda-cpp
- [ ] Feed your own data inflow for training and finetuning
- [ ] Pruning and Quantization


#### License

Distributed under the GNU General Public License v3.0 License. See `LICENSE.txt` for more information.


#### Contact

Isaac N. Xu - [@email](xuningandy@gmail.com)

