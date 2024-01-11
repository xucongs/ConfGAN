import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
from utils.common import cal_u
from model.gnn import MessagePassing


# Aggregation of node and edge information in the graph
class MergeNodeEdge(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, inputs_shape):
        
        self.dense_1 = layers.Dense(self.units, use_bias=False, activation=layers.LeakyReLU(alpha=0.01))
        self.dense_2 = layers.Dense(self.units, use_bias=False, activation=layers.LeakyReLU(alpha=0.01))
        self.dense_3 = layers.Dense(self.units/2, use_bias=False, activation=layers.LeakyReLU(alpha=0.01))
        self.dense_4 = layers.Dense(self.units/2, use_bias=False, activation=layers.LeakyReLU(alpha=0.01))
        self.cat  =  layers.Concatenate(axis=-1)
        
    def call(self, inputs):
        node_attr, edge_attr, pair_indices = inputs        
        node_attr_1 = tf.gather(node_attr, pair_indices[:, 0], axis=0)
        node_attr_2 = tf.gather(node_attr, pair_indices[:, 1], axis=0)
        
        node_attr = self.cat([self.dense_1(node_attr_1), self.dense_2(node_attr_2)])
        node_attr = self.dense_3(node_attr)
        edge_attr = self.dense_4(edge_attr)
        
        return self.cat([node_attr, edge_attr ])

# generator
def make_generator_model(units = 64, mpl_num = 6):
    inputs = {
        'atom_features': (32, 'float32'),
        'bond_features': (8, 'float32'),
        'pair_indices': (2, 'int32'),
        'atoms_mofit_indices': ((), 'int32'),
        'atoms_mofit_values': ((), 'int32'),
        'pair_motif_indices': (2, 'int32'),
        'motif_bonds': (8, 'float32'),
        'motif_node': (32, 'float32'),
        'node_indices': ((), 'int32'),
        'motif_indices': ((), 'int32', True)
    }

    inputs_layers = {}
    for key, value in inputs.items():
        inputs_layers[key] = layers.Input(value[0], dtype=value[1], name = key)

    motif_node_update = layers.Dense(units*2, use_bias=False, activation=layers.LeakyReLU(alpha=0.01))(inputs_layers['motif_node'])
    motif_node_update = tf.RaggedTensor.from_row_splits(
        values=motif_node_update, row_splits=inputs_layers['atoms_mofit_indices']
    )
    
    motif_node_update = tf.reduce_sum(motif_node_update, axis=1)
    motif_node_mp = MessagePassing(units, 4)([motif_node_update, inputs_layers['motif_bonds'], inputs_layers['pair_motif_indices']])
    motif_node_repeat = tf.repeat(motif_node_mp, repeats=inputs_layers['atoms_mofit_indices'][1:] - inputs_layers['atoms_mofit_indices'][0:-1], axis=0)
    motif_node_sort = tf.gather(motif_node_repeat, tf.argsort(inputs_layers['atoms_mofit_values']))
    motif_node_gat = tf.gather(motif_node_sort, inputs_layers['pair_indices'][:, 0], axis=0)

    noise = tf.random.normal([tf.shape(inputs_layers['bond_features'])[0], units])

    atom_features_update = MessagePassing(units, 4)([inputs_layers['atom_features'], inputs_layers['bond_features'], inputs_layers['pair_indices']])
    label = MergeNodeEdge(units)([atom_features_update, inputs_layers['bond_features'], inputs_layers['pair_indices']])

    joined = layers.multiply([noise, label])

    gen_dis = layers.Dense(units*2, use_bias=False, activation=layers.LeakyReLU(alpha=0.01))(joined)
    
    #Concatenate each atom in the molecular motif with the corresponding position in the molecular graph
    gen_dis = layers.Concatenate(axis=-1)([motif_node_gat, gen_dis])

    for _ in range(mpl_num):
        gen_dis = layers.Dense(units*2, use_bias=False, activation=layers.LeakyReLU(alpha=0.01))(gen_dis)

    gen_dis = layers.Dense(1, use_bias=False, activation=layers.LeakyReLU(alpha=0.01))(gen_dis)

    model = keras.Model(
        inputs=list(inputs_layers.values()),
        outputs=[gen_dis],
        name='generator'
    )

    return model

# discriminator
def make_discriminator_model(units = 64, mpl_num = 6):

    inputs = {
        'atom_features': (32, 'float32'),
        'bond_features': (8, 'float32'),
        'pair_indices': (2, 'int32'),
        'atoms_mofit_indices': ((), 'int32'),
        'atoms_mofit_values': ((), 'int32'),
        'pair_motif_indices': (2, 'int32'),
        'motif_bonds': (8, 'float32'),
        'motif_node': (32, 'float32'),
        'node_indices': ((), 'int32'),
        'motif_indices': ((), 'int32', True),
        'distance': (1, 'float32')
    }

    inputs_layers = {}
    for key, value in inputs.items():
        inputs_layers[key] = layers.Input(value[0], dtype=value[1], name = key)

    bond_features_update = MessagePassing(units, 4)([inputs_layers['atom_features'], inputs_layers['bond_features'], inputs_layers['pair_indices']])
    atoms_edge_merge = MergeNodeEdge(units)([bond_features_update, inputs_layers['bond_features'], inputs_layers['pair_indices']])
    distance_reshape = layers.Dense(units, use_bias=False, activation=layers.LeakyReLU(alpha=0.01))(inputs_layers['distance'])

    atmos_dis_merge = layers.Concatenate(axis=-1)([atoms_edge_merge, distance_reshape])

    motif_node_update = layers.Dense(units*2, use_bias=False, activation=layers.LeakyReLU(alpha=0.01))(inputs_layers['motif_node'])
    motif_node_update = tf.RaggedTensor.from_row_splits(values=motif_node_update, row_splits=inputs_layers['atoms_mofit_indices'])
    motif_node_update = tf.reduce_sum(motif_node_update, axis=1)
    motif_node_mp = MessagePassing(units, 4)([motif_node_update, inputs_layers['motif_bonds'], inputs_layers['pair_motif_indices']])
    motif_node_repeat = tf.repeat(motif_node_mp, repeats=inputs_layers['atoms_mofit_indices'][1:] - inputs_layers['atoms_mofit_indices'][0:-1], axis=0)
    motif_node_sort = tf.gather(motif_node_repeat, tf.argsort(inputs_layers['atoms_mofit_values']))
    motif_node_gat = tf.gather(motif_node_sort, inputs_layers['pair_indices'][:, 0], axis=0)

    dis = layers.Dense(units*2, use_bias=False, activation=layers.LeakyReLU(alpha=0.01))(atmos_dis_merge)
    dis = layers.Concatenate(axis=-1)([motif_node_gat, dis])

    for _ in range(mpl_num):
        dis = layers.Dense(units*2, use_bias=False, activation=layers.LeakyReLU(alpha=0.01))(dis)

    dis = layers.Dense(1)(dis)

    model = keras.Model(
        inputs=list(inputs_layers.values()),
        outputs=[dis],
        name='discriminator'
    )

    return model





















