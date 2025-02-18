A refactor of DP5_rewrite. Namespace structure implemented for legibility and maintainability. 

To get started:
- clone this repository
- navigate to the folder
- install via `pip install -e .`

## Notes for `sgnn` branch
Install dependencies using `pip install -e .`, then `pip install pytorch==2.2.1
dgl==2.1.0.cu118 dgllife==0.3.2`, then downgrade `numpy` using `pip install
numpy==1.26.4`.

Right now, the model isn't loading correctly from the checkpoint. It loads fine
using the inference script in the SGNN repo, so I'm not sure why it's not
loading here.

Clone `SGNN` conda env, then install DP5 with `pip install -e .`.