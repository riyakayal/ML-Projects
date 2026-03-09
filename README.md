# ML-Projects
![alt text](https://github.com/riyakayal/ML-Projects/blob/main/images/HPC%20ML.png?raw=true)
## A collection of HPC scale Molecular Machine Learning codes from scratch in python with different state-of-the-art techniques, fully CPU parallelised.
### Requirements:
* Here are all the requirements:
```bash
pip install pandas numpy torch rdkit scikit-learn scipy matplotlib psutil
```
* If you're on a system where RDKit isn't available via pip directly (older environments), use:
```bash
pip install pandas numpy torch scikit-learn scipy matplotlib psutil
conda install -c conda-forge rdkit
```
* And if you're on a **shared HPC cluster** (for path like `/usr/users/username/miniconda3/`), you likely need:
```bash
pip install pandas numpy torch scikit-learn scipy matplotlib psutil --break-system-packages
conda install -c conda-forge rdkit
```

