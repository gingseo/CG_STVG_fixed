# CG_STVG_fixed
fixed errors and warnings code..

model_zoo
https://github.com/HengLan/CGSTVG/tree/main?tab=readme-ov-file
에서 레즈넷, vidswin은 받고, roberta는
transformers-cli download roberta-base --cache-dir /workspace/CGSTVG/model_zoo/
로 다운.
mkdir -p /workspace/CGSTVG/model_zoo
cd /workspace/CGSTVG/model_zoo
huggingface-cli download roberta-base --local-dir roberta-base
./train.sh 실행시 학습됨.

원본 코드: https://github.com/HengLan/CGSTVG/tree/main?tab=readme-ov-file
데이터셋: https://intxyz-my.sharepoint.com/personal/zongheng_picdataset_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzongheng%5Fpicdataset%5Fcom%2FDocuments%2FHC%2DSTVG&ga=1
![alt text](image.png)
밖의 json을 사용, 이름을 test, train, query로 변경
파일 중 data/hc-stvg2/annos/hcstvg_v2/test_sub.json은 내가 inference용으로 따로 뺴둔거

 ![alt text](image-1.png)

 pip list
 Package                  Version
------------------------ -----------
absl-py                  2.1.0
asttokens                3.0.0
boto3                    1.37.15
botocore                 1.37.15
certifi                  2025.1.31
charset-normalizer       3.4.1
cmake                    3.31.6
comm                     0.2.2
contourpy                1.3.1
cycler                   0.12.1
Cython                   3.0.12
debugpy                  1.8.12
decorator                5.1.1
easydict                 1.13
einops                   0.8.1
exceptiongroup           1.2.2
executing                2.2.0
ffmpeg                   1.4
ffmpeg-python            0.2.0
filelock                 3.17.0
fonttools                4.56.0
fsspec                   2025.2.0
ftfy                     6.3.1
future                   1.0.0
grpcio                   1.71.0
huggingface-hub          0.28.1
idna                     3.10
inquirerpy               0.3.4
ipykernel                6.29.5
ipython                  8.32.0
ipywidgets               8.1.5
jedi                     0.19.2
Jinja2                   3.1.4
jmespath                 1.0.1
jupyter_client           8.6.3
jupyter_core             5.7.2
jupyterlab_widgets       3.0.13
kiwisolver               1.4.8
lit                      18.1.8
Markdown                 3.7
MarkupSafe               2.1.5
matplotlib               3.10.1
matplotlib-inline        0.1.7
mpmath                   1.3.0
nest-asyncio             1.6.0
networkx                 3.3
numpy                    1.23.5
nvidia-cublas-cu11       11.10.3.66
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-cupti-cu11   11.7.101
nvidia-cuda-cupti-cu12   12.1.105
nvidia-cuda-nvrtc-cu11   11.7.99
nvidia-cuda-nvrtc-cu12   12.1.105
nvidia-cuda-runtime-cu11 11.7.99
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu11        8.5.0.96
nvidia-cudnn-cu12        8.9.2.26
nvidia-cufft-cu11        10.9.0.58
nvidia-cufft-cu12        11.0.2.54
nvidia-curand-cu11       10.2.10.91
nvidia-curand-cu12       10.3.2.106
nvidia-cusolver-cu11     11.4.0.1
nvidia-cusolver-cu12     11.4.5.107
nvidia-cusparse-cu11     11.7.4.91
nvidia-cusparse-cu12     12.1.0.106
nvidia-nccl-cu11         2.14.3
nvidia-nccl-cu12         2.20.5
nvidia-nvjitlink-cu12    12.1.105
nvidia-nvtx-cu11         11.7.91
nvidia-nvtx-cu12         12.1.105
opencv-python            4.11.0.86
packaging                21.3
parso                    0.8.4
pexpect                  4.9.0
pfzy                     0.3.4
pillow                   11.0.0
pip                      22.0.2
platformdirs             4.3.6
prompt_toolkit           3.0.50
protobuf                 6.30.1
psutil                   6.1.1
ptyprocess               0.7.0
pure_eval                0.2.3
Pygments                 2.19.1
pyparsing                3.2.1
python-dateutil          2.9.0.post0
pytorch-pretrained-bert  0.6.2
PyYAML                   6.0.2
pyzmq                    26.2.1
regex                    2024.11.6
requests                 2.32.3
s3transfer               0.11.4
safetensors              0.5.3
scipy                    1.15.2
setuptools               59.6.0
six                      1.17.0
stack-data               0.6.3
sympy                    1.13.1
tensorboard              2.19.0
tensorboard-data-server  0.7.2
timm                     1.0.15
tokenizers               0.21.1
torch                    2.0.1
torchaudio               2.3.1+cu121
torchdata                0.6.1
torchtext                0.15.2
torchvision              0.15.2
tornado                  6.4.2
tqdm                     4.67.1
traitlets                5.14.3
transformers             4.49.0
triton                   2.0.0
typing_extensions        4.12.2
urllib3                  2.3.0
wcwidth                  0.2.13
Werkzeug                 3.1.3
wheel                    0.45.1
widgetsnbextension       4.0.13
yacs                     0.1.8
