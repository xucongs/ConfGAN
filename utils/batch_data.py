import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#processing of batch data

def batch_merge(data_dict):
    """

    Merge the batch number of graphs into a graph

    graph shape : atom_features :  [batch, node_row_len, node_col]   --> [node_row_len*batch, node_col]
                  bond_features :  [batch, edge_row_len, edge_col]   --> [edge_row_len*batch, edge_col]
                  pair_indices : [batch, edge_row_len, 2]   --> [[merge_dim(pair_indices...)+[0+atom_features[0].shape[1]+node_attr[1].shape[1]+.....node_attr[batch-1].shape[0]]].shape[0], 2]
                  y_batch  :   [batch, y_batch_row_len, 1]   --> [y_batch_row_len*batch, 1]
    """
    
    node_row_len = data_dict['atom_features'].row_lengths()
    edge_row_len = data_dict['bond_features'].row_lengths()
    
    motif_row_len = data_dict['atoms_mofit'].row_lengths()
    motif_edge_row_len = data_dict['pair_motif_indices'].row_lengths()
    data_dict['node_indices'] = tf.repeat(tf.range(len(node_row_len)), node_row_len)
    data_dict['motif_indices'] = tf.repeat(tf.range(len(motif_row_len)), motif_row_len)

    gather_indices = tf.repeat(tf.range(len(node_row_len))[:-1], edge_row_len[1:])
    motif_gather_indices = tf.repeat(tf.range(len(motif_row_len))[:-1], motif_edge_row_len[1:])

    pair_add = tf.cumsum(node_row_len[:-1])
    pair_add = tf.cast(tf.pad(tf.gather(pair_add, gather_indices), [(edge_row_len[0], 0)]), dtype=tf.int32)
    pair_motif_add = tf.cumsum(motif_row_len[:-1])
    pair_motif_add = tf.cast(tf.pad(tf.gather(pair_motif_add, motif_gather_indices), [(motif_edge_row_len[0], 0)]),
                             dtype=tf.int32)

    atoms_motif_adds, _ = tf.unique(pair_add)
    atoms_motif_adds = tf.reshape(atoms_motif_adds, [-1, 1, 1])
    data_dict['atoms_mofit'] = atoms_motif_adds + data_dict['atoms_mofit']
    data_dict['atoms_mofit'] = data_dict['atoms_mofit'].merge_dims(0, 1)

    data_dict['atoms_mofit_indices']  =  data_dict['atoms_mofit'].row_splits
    data_dict['atoms_mofit_values'] = data_dict['atoms_mofit'].values
    

    data_dict['pair_indices'] = data_dict['pair_indices'].merge_dims(0, 1).to_tensor() + pair_add[:, tf.newaxis]


    data_dict['atom_features'] = data_dict['atom_features'].merge_dims(0, 1).to_tensor()
    data_dict['bond_features'] = data_dict['bond_features'].merge_dims(0, 1).to_tensor()
    
    data_dict['motif_bonds'] = data_dict['motif_bonds'].merge_dims(0, 1).to_tensor()
    data_dict['motif_node'] = tf.gather(data_dict['atom_features'], data_dict['atoms_mofit_values'], axis=0)
    data_dict['pair_motif_indices'] = data_dict['pair_motif_indices'].merge_dims(0, 1).to_tensor() + pair_motif_add[:, tf.newaxis]

    data_dict['distance'] = tf.reshape(data_dict['distance'].merge_dims(outer_axis=0, inner_axis=1), (-1, 1))
