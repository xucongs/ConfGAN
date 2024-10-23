import pickle
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import random
import tensorflow as tf
from rdkit.Chem import AllChem
from mol2graph import graphs_from_mols
from tqdm import tqdm
import time
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def process_dataset(dataset_type, max_batch_size):
    if dataset_type == "QM9":
        path = r'/home/xucongs/work/data/rdkit_folder/qm9'
        files = os.listdir(path)
        output_files = ["qm9_train.pkl", "qm9_val.pkl", "qm9_test.pkl"]
    elif dataset_type == "Drugs":
        path = r'/home/xucongs/work/data/rdkit_folder/drugs'
        files = [i for i in os.listdir(path) if i.split('.')[-1] == 'pickle']
        output_files = ["drugs_train.pkl", "drugs_val.pkl", "drugs_test.pkl"]
    else:
        raise ValueError("Unsupported dataset type. Choose 'QM9' or 'Drugs'.")

    start = time.time()
    datas = []
    current_batch = []

    # Process files based on the dataset type
    if dataset_type == "QM9":
        for file in tqdm(files, desc="Processing QM9 files"):
            try:
                with open('/'.join([path, file]), 'rb') as r:
                    data = pickle.load(r)
                    mols = [mol_dict['rd_mol'] for mol_dict in data['conformers']]
                    current_batch.extend(mols)
                    while len(current_batch) >= max_batch_size:
                        datas.append(graphs_from_mols(current_batch[:max_batch_size]))
                        current_batch = current_batch[max_batch_size:]
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    elif dataset_type == "Drugs":
        processed_files = 0
        for file in tqdm(files, desc="Processing Drugs files"):
            if processed_files >= 62500:
                break
            try:
                with open(path + '/' + file, 'rb') as r:
                    confs = pickle.load(r)['conformers']
                    mols = []
                    for conf in confs:
                        mol = conf['rd_mol']
                        mol = AllChem.RemoveAllHs(mol)
                        atom_num = len([atom.GetAtomicNum() for atom in mol.GetAtoms()])
                        if atom_num < 2 or atom_num > 40:
                            continue
                        mols.append(mol)
                    if mols:  # Count only if there are valid molecules
                        processed_files += 1
                        current_batch.extend(mols)
                        while len(current_batch) >= max_batch_size:
                            datas.append(graphs_from_mols(current_batch[:max_batch_size]))
                            current_batch = current_batch[max_batch_size:]
                            
            except Exception as e:
                print(f"Error processing file {file}: {e}")

            tf.keras.backend.clear_session()

    random.shuffle(datas)
    
    # Split data into train, validation, and test sets
    train_size = int(0.8 * len(datas))
    val_size = int(0.1 * len(datas))
    test_size = len(datas) - train_size - val_size

    train_data = datas[:train_size]
    val_data = datas[train_size:train_size + val_size]
    test_data = datas[train_size + val_size:]

    os.makedirs('qm9', exist_ok=True)
    os.makedirs('drugs', exist_ok=True)

    if dataset_type == "QM9":
        output_dir = 'qm9'
    elif dataset_type == "Drugs":
        output_dir = 'drugs'

    with open(os.path.join(output_dir, output_files[0]), 'wb') as w:
        pickle.dump(train_data, w)
    with open(os.path.join(output_dir, output_files[1]), 'wb') as w:
        pickle.dump(val_data, w)
    with open(os.path.join(output_dir, output_files[2]), 'wb') as w:
        pickle.dump(test_data, w)

    end = time.time()
    result = end - start
    print(f"{dataset_type} dataset processing time: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process QM9 or Drugs dataset.")
    parser.add_argument('--dataset', type=str, required=True, help="Specify the dataset to process: 'QM9' or 'Drugs'")
    parser.add_argument('--batch_size', type=int, default=128, help="Specify the maximum batch size")
    args = parser.parse_args()

    process_dataset(args.dataset, args.batch_size)
