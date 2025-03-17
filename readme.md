A refactor of DP5_rewrite. Namespace structure implemented for legibility and maintainability. 

To get started:
- clone this repository
- navigate to the folder
- install via `pip install -e .`

## Instructions for setting up environment with SGNN dependencies installed
Install `torch`, `dgl` and `dgllife` using `pip` before installing the package.
```
conda create -n dp5_new python==3.11
conda activate dp5_new
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html
pip install dgllife==0.3.2
pip install -e .
```

If this is not done, the SGNN model will not be available.

## Notes for `snap-nmr` branch
This branch is a modification of the main branch, to adapt pydp4 to the specific
requirements of snap-nmr.