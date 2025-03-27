# PO3AD
## [CVPR 2025] PO3AD: Predicting Point Offsets toward Better 3D Point Cloud Anomaly Detection
## Environments
We run our code on RTX3090 with Python 3.8 and PyTorch 1.9.0.

**Create Conda Environment**
```
conda create -n PO3AD python=3.8
conda activate PO3AD
```

**Install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)**
```
conda install -c pytorch -c nvidia -c conda-forge pytorch=1.9.0 cudatoolkit=11.1 torchvision
conda install openblas-devel -c anaconda

# Uncomment the following line to specify the cuda home. Make sure `$CUDA_HOME/nvcc --version` is 11.X
# export CUDA_HOME=/usr/local/cuda-11.1
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# Or if you want local MinkowskiEngine
cd lib
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```
## Dataset Preparation
Download the [AnomalyShapeNet](https://github.com/Chopper-233/Anomaly-ShapeNet) and [Real3D-AD](https://github.com/M-3LAB/Real3D-AD) datasets.
Put the data in the corresponding folders. For example:
```
PO3AD
├── datasets
│   ├── AnomalyShapeNet
│   │   ├── dataset
│   │   │   ├── obj
│   │   │   ├── pcd
```
## Training & Evaluation
(1) Training
```
python train.py --dataset AnomalyShapeNet --category ashtray0
```

(2) Evaluation (We provide checkpoints in [Google Drive](https://drive.google.com/drive/folders/14UU14Tl1NbogS1S4yRM2MbgoF4YG-NWO?usp=sharing).)
```
python eval.py --dataset AnomalyShapeNet --category ashtray0 --checkpoint_name ashtray0.pth
```

## Citation
If you find this project helpful for your research, please consider citing the following BibTex entry:
```
@inproceedings{PO3AD,
  title={PO3AD: Predicting Point Offsets toward Better 3D Point Cloud Anomaly Detection},
  author={Ye, Jianan and Zhao, Weiguang and Yang, Xi and Cheng, Guangliang and Huang, Kaizhu},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
