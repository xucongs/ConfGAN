# ConfGAN 
![ConfGAN](ConfGAN.gif)

## Requirements

* python (version>=3.8.0)
* tensorflow-gpu (version>=2.6.0)
* rdkit (version>=2023.03.1)
* numpy (version==1.20.3)
* networkx (version>=3.1)


## Installation


Create a conda environment named `ConfGAN` from `env.yml` :

```bash
conda env create --file env.yml
```



## Data

Download the GEOM raw dataset: [here](https://dataverse.harvard.edu/api/access/datafile/4327252).

Process raw data using the command: `python ./data/process_data.py --dataset QM9/Drugs`

The processed data can be downloaded from here: [Google Drive](https://drive.google.com/file/d/1kAi9I2OTOf_W6rlcI87JpKvf3qrsmbLc/view?usp=sharing) .


## Usage

### Generate Conformations

Generating conformations is very simple and can be done using three methods. 

The first method is using mol files.
```bash
python main.py --generate --mol test.mol --num_conf 50
```
The second method is to input SMILES
```bash
python main.py --generate --smiles 'CNC(=O)c1nc(C)cs1' --num_conf 50
```
The third method is to use a CSV file for batch generation, 
where the CSV file format: 
smiles, num_conf
'CNC(=O)c1nc(C)cs1', 50
'CNC(=O)c1nc(N)cs1', 20

```bash
python main.py --generate True --csv test.csv
```


More generation options can be found in `generate.py`.

### Train

Example: training a model for QM9 molecules.

```bash
python main.py --train --train_dataset ./data/qm9_train.pkl --val_dataset ./data/qm9_val.pkl
```

More training options can be found in `main.py`.

## Citation
## Contact

<12049045@mail.sustech.edu.cn> 

## Updates

- Oct. 19, 2024. 



