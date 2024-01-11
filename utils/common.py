import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from rdkit.Chem import AllChem
from data.mol2graph import graphs_from_mols


# bond energy
def bond_energy(r, k, req):
    return 0.5 * k * (r - req) ** 2

# vdw energy
def vdW_energy(r, A, B):
    return B * ((A / r)**12 - 2*(A / r)**6)


    
# Calculate pseudopotential energy.    
def cal_u(dis, u_parm):
    u_parm = u_parm.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_mask = u_parm[:, -1] < 0.5
    vdW_mask = ~bond_mask
    bond_mask = tf.cast(bond_mask, tf.float32)
    vdW_mask = tf.cast(vdW_mask, tf.float32)
    u = tf.reshape(bond_mask * bond_energy(dis[:, 0], u_parm[:, 0], u_parm[:, 1]) + vdW_energy(dis[:, 0], u_parm[:, 0], u_parm[:, 1]) * vdW_mask, (-1, 1))
    return u
    
def mol2g(mol, num_conf, is_train = False):    
    try:
        mol = AllChem.RemoveHs(mol)
    except:
        pass   
    graphs = graphs_from_mols([mol] * num_conf, is_train = False )
    return graphs, mol