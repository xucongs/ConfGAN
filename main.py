import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import warnings
import tensorflow as tf
import pickle
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from rdkit.Chem import AllChem
import pandas as pd
from run import GANTrainer, GANGenerater
from model.confgan import make_generator_model, make_discriminator_model
from utils.batch_data import batch_merge
from model.confgan import make_generator_model, make_discriminator_model
from utils.common import mol2g
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=eval, default=True, choices=[True, False])
parser.add_argument('--gpuid', type=str, default='0')

parser.add_argument('--train', type=eval, default=False, choices=[True, False])
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--train_dataset', type=str, default='./data/qm9/train.pkl')
parser.add_argument('--val_dataset', type=str, default='./data/qm9/val.pkl')


parser.add_argument('--generate', type=eval, default=True, choices=[True, False])
parser.add_argument('--use_ff', type=eval, default=True, choices=[True, False])
parser.add_argument('--num_conf', type=int, default=50)
parser.add_argument('--chekpoint_dir', type=str, default='./checkpoint')
parser.add_argument('--mol', type=str, default='')
parser.add_argument('--smiles', type=str, default='')
parser.add_argument('--out_path', type=str, default='./')
parser.add_argument('--out_xyz_file', type=str, default='')
parser.add_argument('--csv', type=str, default='')

parser.add_argument('--num_workers', type=int, default=8)


args = parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid


# Initialize the mode
generator = make_generator_model(args.hidden_dim)
discriminator = make_discriminator_model(args.hidden_dim)
generator_optimizer = RMSprop(learning_rate=args.lr)
discriminator_optimizer = RMSprop(learning_rate=args.lr)
checkpoint_dir = args.chekpoint_dir


# Execute training
if args.train:
    trainer = GANTrainer(generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_dir)
    with open(args.train_dataset, 'rb') as r:
        datasets = pickle.load(r)
    dataset_batch = []
    for _, data_dict in datasets[0].items():
        batch_merge(data_dict)
        dataset_batch.append(data_dict)
        trainer.train(dataset_batch, args.epoch)

    
        
        
#  Generate conformation    
if  args.generate:
    generater = GANGenerater(generator, checkpoint_dir)
    # Input a Mol file 
    if args.mol:
        mol =  AllChem.MolFromMolFile(args.mol)
        graphs, mol = mol2g(mol, num_conf = args.num_conf)
        batch_merge(graphs)
        smiles = AllChem.MolToSmiles(mol)
        if not args.out_xyz_file:
            out_xyz_file = smiles+'.xyz'
        else: out_xyz_file = args.out_xyz_file
        generater.get_conf(mol, graphs, out_path = args.out_path, xyz_file = out_xyz_file, use_ff = args.use_ff)
        print(f"{smiles} with {args.num_conf} conformations generated, completed.")
        
    # Input a smiles string    
    if args.smiles:
        mol =  AllChem.MolFromSmiles(args.smiles)
        graphs, mol = mol2g(mol, num_conf = args.num_conf)
        batch_merge(graphs)
        if not args.out_xyz_file:
            out_xyz_file = AllChem.MolToSmiles(mol)+'.xyz'
        else: out_xyz_file = args.out_xyz_file
        generater.get_conf(mol, graphs, out_path = args.out_path, xyz_file = out_xyz_file, use_ff = args.use_ff)
        print(f"{args.smiles} with {args.num_conf} conformations generated, completed.")
        
    # Input a CSV file for batch generation 
    if args.csv:
        df = pd.read_csv(args.csv)
        for _, row in df.iterrows():
            mol =  AllChem.MolFromSmiles(row['smiles'])
            graphs, mol = mol2g(mol, num_conf = row['num_conf'])
            batch_merge(graphs)
            out_xyz_file = row['smiles']+'.xyz'
            generater.get_conf(mol, graphs, out_path = args.out_path, xyz_file = out_xyz_file, use_ff = args.use_ff)
            print(f"{row['smiles']} with {row['num_conf']} conformations generated, completed.")