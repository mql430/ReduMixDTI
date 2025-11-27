# ReduMixDTI
ReduMixDTI: prediction of drug-target interaction with feature redundancy reduction and interpretable attention mechanism

# Cite
```
@article{liu2024ReduMixDTI,
  title = {ReduMixDTI: Prediction of Drug--Target Interaction with Feature Redundancy Reduction and Interpretable Attention Mechanism},
  author = {Liu, Mingqing and Meng, Xuechun and Mao, Yiyang and Li, Hongqi and Liu, Ji},
  year = 2024,
  month = dec,
  journal = {Journal of Chemical Information and Modeling},
  volume = {64},
  number = {23},
  pages = {8952--8962},
  publisher = {American Chemical Society},
  issn = {1549-9596, 1549-960X},
  doi = {10.1021/acs.jcim.4c01554},
  langid = {english}
}
```


## Setup
Setting up for this project involves installing dependencies and preparing the datasets.

### Installing dependencies
```
dgl==2.2.1+cu118
dgllife==0.3.2
einops==0.8.0
numpy==1.26.3
pandas==2.2.2
prefetch_generator==1.0.3
prettytable==3.10.0
scikit_learn==1.5.0
torch==2.2.1
tqdm==4.65.0
yacs==0.1.8
```

### Preparing Dataset
The datasets folder contains all experimental data utilized in ReduMixDTI, including the BindingDB [1], BioSNAP [2], and Human [3] datasets. These datasets are provided in CSV format and consist of three columns: **SMILES**, **Protein** and **Y**.

## Run
### For Human dataset
```bash
$ python main.py --cfg ./configs/human.yaml --outname human_model --data human --num_worker 0
```
### For BioSNAP dataset
```
$ python main.py --cfg ./configs/biosnap.yaml --outname biosnap_model --data biosnap --num_worker 0
```
### For BindingDB dataset
```
$ python main.py --cfg ./configs/bindingdb.yaml --outname bindingdb_model --data bindingdb --num_worker 0
```

## Reference
>[1] Bai, P.; Miljković, F.; Ge, Y.; Greene, N.; John, B.; Lu, H. Hierarchical Clustering Split for Low-Bias Evaluation of Drug-Target Interaction Prediction. 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). 2021; pp 641–644.  
>
>[2] Zitnik, M.; Sosic, R.; Leskovec, J. BioSNAP Datasets: Stanford Biomedical Network Dataset Collection. http://snap.stanford.edu/biodata 2018, 5.
>
>[3] Liu, H.; Sun, J.; Guan, J.; Zheng, J.; Zhou, S. Improving Compound–Protein Interaction Prediction by Building up Highly Credible Negative Samples. Bioinformatics 2015, 31, i221–i229.
