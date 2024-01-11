import os
import time
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import RMSprop
from tensorflow.train import Checkpoint
from utils.common import cal_u
from rdkit.Chem import AllChem
from rdkit import Chem

class GANTrainer:
    def __init__(self, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_dir):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def train_step(self, data_dict):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_dis = self.generator(data_dict, training=True)
            real_u = data_dict['distance']
            '''
            During training, it is recommended to initially exclude the calculation of potential energy and 
            only focus on calculating distances, as unrealistic initial distances can result in infinite potential energy
            '''
            #real_u = cal_u(data_dic['distance'], data_dic['node_indices'], data_dic['pair_indices'], data_dic['u_parm'])
            #fake_u = cal_u(gen_dis, data_dic['node_indices'], data_dic['pair_indices'], data_dic['u_parm'])
            
            #data_dict['distance'] =  real_u
            real_output = self.discriminator(data_dict, training=True)

            data_dict['distance'] = gen_dis
            fake_output = self.discriminator(data_dict, training=True)
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output, real_u, data_dict, self.discriminator)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, datasets, epochs):
        for epoch in range(epochs):
            start = time.time()
            for dataset in datasets:
                self.train_step(dataset)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    @staticmethod
    def generator_loss(fake_output):
        return -tf.reduce_mean(fake_output)

    def discriminator_loss(self, real_output, fake_output, real_u, data_dict, discriminator):
        dis_indices = tf.gather(data_dict['node_indices'], data_dict['pair_indices'], axis=0)
        gp_loss = self.gradient_penalty(real_u, real_u.shape[0], data_dict, discriminator)
        real_output = tf.math.unsorted_segment_sum(
            real_output,
            dis_indices[:, 0],
            num_segments=dis_indices[-1, 0] + 1,
        )
        fake_output = tf.math.unsorted_segment_sum(
            fake_output,
            dis_indices[:, 0],
            num_segments=dis_indices[-1, 0] + 1
        )
        real_loss = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)
        real_loss = real_loss + 0.01 * gp_loss
        return real_loss

    @staticmethod
    def gradient_penalty(real_u, batch_size, data_dict, discriminator):
        alpha = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=1.)
        differences = data_dict['distance'] - real_u
        interpolates = real_u + (alpha * differences)
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            data_dict['distance'] = interpolates
            pred = discriminator(data_dict)
        gradients = tape.gradient(pred, [interpolates])

        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=-1))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        return gradient_penalty
   
class GANGenerater:
    def __init__(self, generator, checkpoint_dir):
        self.generator = generator
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = Checkpoint(generator=generator)
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        
    def get_pos(self, data_dict, pred_dis):
        position = tf.random.normal([data_dict['atom_features'].shape[0], 3], 0, 1, dtype=tf.float32)
        position = tf.Variable(position, dtype=tf.float32)
        initial_learning_rate = 1.0
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate ,
            decay_steps=1000,
            decay_rate=0.1,
            staircase=True
        )       
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        iter_distance = None
        pred_dis = tf.reshape(pred_dis, [-1])   
        
        for n in range(0,3000):   
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule(n))
            with tf.GradientTape() as tape:
                pos1 = tf.gather(position, data_dict['pair_indices'][:, 0], axis=0)
                pos2 = tf.gather(position, data_dict['pair_indices'][:, 1], axis=0)        
                iter_distance = tf.norm(pos1-pos2, axis = -1)      
                if n >= 2000:
                    loss = tf.reduce_mean(tf.square(iter_distance - pred_dis ) + tf.reshape(cal_u(tf.reshape(iter_distance, [-1, 1]), data_dict['u_parm']), [-1] ) )
 
                else: 
                    loss = tf.reduce_mean(tf.square(iter_distance - pred_dis ))  
            grads = tape.gradient(loss, [position])
            optimizer.apply_gradients(grads_and_vars=zip(grads, [position]))
        position = tf.dynamic_partition(position, data_dict['node_indices'], data_dict['node_indices'][-1]+1)
        return position
    
    def get_conf(self, mol, graphs, out_path, xyz_file, use_ff = True):  
        with open(os.path.join(out_path, xyz_file), 'w') as w:
            pred_dis= self.generator(graphs)
            position = self.get_pos(graphs, pred_dis)        
            smiles = AllChem.MolToSmiles(mol, isomericSmiles=False)
            
            for pos in position:           
                conf = Chem.Conformer()
                mol_new= AllChem.MolFromSmiles(smiles)
                sub = mol.GetSubstructMatch(mol_new)
                for k, j in enumerate(sub):
                    conf.SetAtomPosition(k, (float(pos[j][0]), float(pos[j][1]), float(pos[j][2])))
                mol_new.AddConformer(conf)      
                mol_new = AllChem.AddHs(mol_new, addCoords = True)
                # force field
                if use_ff:
                    AllChem.MMFFOptimizeMolecule(mol_new, mmffVariant='MMFF94s')
                w.write(AllChem.MolToXYZBlock(mol_new))