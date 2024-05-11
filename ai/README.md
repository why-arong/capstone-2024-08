## environment
```bash
conda create -n loro python=3.10
conda activate loro

cd vits2_pytorch
sudo apt-get install espeak
sudo pip3 install --ignore-installed -r requirements.txt
```

## vscode에서 Jupyter notebook 사용하는 경우
```bash
pip3 install jupyter
pip3 install ipykernel
python3 -m ipykernel install --user --name loro --display-name loro
```


## dataset
뉴스 대본 및 앵커 음성 데이터셋 다운로드
- https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71557

└── ai
    └── dataset
        ├── VL
        │   └── SPK003
        │       └── SPK003YTNSO162
        │           └── SPK003YTNSO162F001.json
        └── VS
            └── SPK003
                └── SPK003YTNSO162
                    └── SPK003YTNSO162F001.wav
    

```bash
ln -s /path/to/dataset/VL DUMMY3
python3 preprocess_filelist.py 

```