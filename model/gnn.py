import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.node_dim = input_shape[0][-1]
        self.edge_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.edge_dim, self.node_dim * self.node_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.node_dim * self.node_dim), initializer="zeros", name="bias",
        )
        self.built = True

    def call(self, inputs):
        node_attr, edge_attr, pair_indices = inputs


        edge_attr = tf.matmul(edge_attr, self.kernel) + self.bias
        edge_attr = tf.reshape(edge_attr, (-1, self.node_dim, self.node_dim))

        node_attr_neighbors = tf.gather(node_attr, pair_indices[:, 1])
        node_attr_neighbors = tf.expand_dims(node_attr_neighbors, axis=-1)

        transformed_attr = tf.matmul(edge_attr, node_attr_neighbors)
        transformed_attr = tf.squeeze(transformed_attr, axis=-1)
        aggregated_node_attr = tf.math.unsorted_segment_sum(
            transformed_attr,
            pair_indices[:, 0],
            num_segments=tf.shape(node_attr)[0],
        )
        return aggregated_node_attr


class MessagePassing(layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.node_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.node_dim)
        self.update_step = layers.GRUCell(self.node_dim + self.pad_length)
        self.built = True

    def call(self, inputs):
        node_attr, edge_attr, pair_indices = inputs
        node_attr_updated = tf.pad(node_attr, [(0, 0), (0, self.pad_length)])

        for i in range(self.steps):
            node_attr_aggregated = self.message_step(
                [node_attr_updated, edge_attr, pair_indices]
            )

            node_attr_updated, _ = self.update_step(
                node_attr_aggregated, node_attr_updated
            )
        return node_attr_updated