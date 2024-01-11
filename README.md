# ConfGAN 
![ConfGAN](ConfGAN.gif)

## Installation


Create a conda environment named `ConfGAN` from `env.yml` :

```bash
conda env create --file env.yml
```


## Data

The  datasets are available [here](coming soon).


## Usage

### Generate Conformations

Generating conformations is very simple and can be done using three methods. 

The first method is using mol files.
```bash
python main.py --mol test.mol --num_conf 50
```
The second method is to input SMILES
```bash
python main.py --smiles 'CNC(=O)c1nc(C)cs1' --num_conf 50
```
The third method is to use a CSV file for batch generation, 
where the CSV file format: 
smiles, num_conf
'CNC(=O)c1nc(C)cs1', 50
'CNC(=O)c1nc(N)cs1', 20

```bash
python main.py --csv test.csv
```


More generation options can be found in `generate.py`.

### Train

Example: training a model for QM9 molecules.

```bash
python train.py --train  True --train_dataset ./data/train.pkl
```

More training options can be found in `train.py`.

## Citation
## Contact

<12049045@mail.sustech.edu.cn> 

## Updates

- Dec 1, 2023. 


