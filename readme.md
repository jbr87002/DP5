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

## Notes for `separate_stages_snapnmr` branch

This branch is a modification of the `snap-nmr` branch, to add the functions to
cache predicted NMR shifts and also reload previously cached shifts to use in a
calculation.

This has been naively implemented by saving the initially calculated shifts
after they are predicted. However, this will not allow dp5 calculations, because
the quantile predictions are re-generated during the course of the dp5
calculation. This probably needs a general refactor, because it doesn't really
make any sense to do the shift prediction twice, I think. For now, we can
implement the caching functions but note that it does not work for dp5
calculations.










