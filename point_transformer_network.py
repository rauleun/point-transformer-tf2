import os
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers

import sampling 
import utils
tf.config.run_functions_eagerly(True)


class point_transformer(tf.keras.layers.Layer):

    def __init__(self, dim=8, attn_hidden = 4, pos_hidden = 8, neighbors = None, name=None, **kwargs):
        super(point_transformer, self).__init__(name=name, **kwargs)
        
        self.neighbors = neighbors
        
        self.initializer = tf.keras.initializers.HeNormal()
        
        self.linear1 = tf.keras.layers.Dense(dim, activation='relu', 
                                             kernel_initializer=self.initializer, 
                                             name='self.linear1')
        self.linear2 = tf.keras.layers.Dense(dim, activation=None, 
                                             kernel_initializer=self.initializer, 
                                             name='self.linear2')
        self.MLP_attn1 =  layers.Dense(attn_hidden, activation='relu',
                                       kernel_initializer=self.initializer,
                                       name='attn_hidden')
        self.MLP_attn2 =  layers.Dense(dim, activation='relu',
                                       kernel_initializer=self.initializer, 
                                       name='self.MLP_attn2')
        self.MLP_pos1 = layers.Dense(pos_hidden, activation='relu',
                                     kernel_initializer=self.initializer, 
                                     name='pos_hidden')
        self.MLP_pos2 = layers.Dense(dim, activation='relu',
                                     kernel_initializer=self.initializer, 
                                     name='self.MLP_pos2')
        self.linear_query = layers.Dense(dim, activation='relu',
                                     kernel_initializer=self.initializer, 
                                     name='self.linear_query')
        self.linear_key = layers.Dense(dim, activation='relu',
                                     kernel_initializer=self.initializer, 
                                     name='self.linear_key')
        self.linear_value = layers.Dense(dim, activation='relu',
                                     kernel_initializer=self.initializer, 
                                     name='self.linear_value')
        


    def call(self, feature, pos):
        
        n = pos.shape[-2]
        
        feature = self.linear1(feature)

        query = self.linear_query(feature)
        key = self.linear_key(feature)
        value = self.linear_value(feature)

        qk = query[:, None, :, :] - key[:, :, None, :]
        pos_rel =  pos[:, None, :, :] - pos[:, :, None, :]

        dist = tf.norm(pos_rel, axis=-1)
        
        if self.neighbors != None:
            __, indices = tf.nn.top_k(-dist, k=self.neighbors)
            
            qk = utils.index_sieve(qk, indices, fix_dim=2)
            pos_rel = utils.index_sieve(pos_rel, indices, fix_dim=2)
            value = utils.index_sieve(value, indices, fix_dim=1)

        pos_emb = self.MLP_pos1(pos_rel)
        pos_emb = self.MLP_pos2(pos_emb)

        mlp_attn1 = self.MLP_attn1(qk + pos_emb)
        attn = tf.nn.softmax(self.MLP_attn2(mlp_attn1), axis=-2)
        out = value * attn
        out = tf.math.reduce_sum(out, axis=-2)
        out = self.linear2(out)

        return out 


class up_sample(tf.keras.layers.Layer):
    
    def __init__(self, dim=8, neighbors=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.neighbors = neighbors
        self.initializer = tf.keras.initializers.HeNormal()
        self.linear = tf.keras.layers.Dense(dim, activation='relu',
                                            kernel_initializer=self.initializer)

    def call(self, feature, pos, pos_up):
        
        num_points_up = pos_up.shape[-2]
        batch_size = pos_up.shape[-3]
        epsilon = 1e-6
        
        local_ind, local_dist = sampling.knn(self.neighbors, pos, pos_up)
        local_weight = 1./(local_dist + epsilon)
        local_weight_norm = tf.math.divide(local_weight, tf.math.reduce_sum(local_weight, axis=-1)[:, :, None])
        local_feature = tf.gather(feature[:, :, :], local_ind[:, :], axis=-2, batch_dims=1)
        local_feature = tf.reshape(local_feature,(-1, local_feature.shape[-3], local_feature.shape[-2], local_feature.shape[-1]))
        new_feature = tf.math.reduce_sum(local_feature*local_weight_norm[:,:,:,None], axis=-2)
        output = self.linear(new_feature)
        
        return output


class down_sample(tf.keras.layers.Layer):

    def __init__(self, dim=8, neighbors=None, sampling_rate=4, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.neighbors = neighbors
        self.sampling_rate = sampling_rate
        self.initializer = tf.keras.initializers.HeNormal()
        self.linear = tf.keras.layers.Dense(dim, activation='relu',
                                            kernel_initializer=self.initializer)

    def call(self, feature, pos):
        num_points = pos.shape[-2]
        batch_size = pos.shape[-3]
        ind, dist = sampling.farthest_point_sampling(num_points//self.sampling_rate, pos)
        sampled_pos = tf.gather(pos[:, :, :], ind[:, :], axis=-2, batch_dims=1)
        
        local_ind, _ = sampling.knn(self.neighbors, pos, sampled_pos)
        local_feature = tf.gather(feature[:, :, :], local_ind[:, :], axis=-2, batch_dims=1)
        local_feature = tf.reshape(local_feature,(-1, local_feature.shape[-3], local_feature.shape[-2], local_feature.shape[-1]))
        output = tf.math.reduce_max(self.linear(local_feature), axis=-2)
        return output, sampled_pos


class MLP(tf.keras.layers.Layer):
    
    def __init__(self, dim=8, dim_out=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.initializer = tf.keras.initializers.HeNormal()
        self.layer1 = tf.keras.layers.Dense(dim, activation='relu',
                                            kernel_initializer=self.initializer)
        self.layer2 = tf.keras.layers.Dense(dim_out, activation=None,
                                            kernel_initializer=self.initializer)
        
    def call(self, feature):
        x1 = self.layer1(feature)
        out = self.layer2(x1)

        return out


class point_transformer_network(tf.keras.Model):

    def __init__(self, num_class, neighbors_sample, neighbors_pt, sampling_rate, batch_size, **kwargs):
        super().__init__(**kwargs)

        self.num_class = num_class
        self.neighbors_sample = neighbors_sample
        self.neighbors_pt = neighbors_pt
        self.sampling_rate = sampling_rate

        self.input_layer = tf.keras.layers.Input(batch_shape=(batch_size, 4096, 9))
        self.MLP1 = MLP(dim=8, dim_out=32, name='self.MLP1')
        self.MLP2 = MLP(dim=8, dim_out=32, name='self.MLP2')
        self.MLP3 = MLP(dim=8, dim_out=self.num_class, name='self.MLP3')

        self.pt0 = point_transformer(dim=8, neighbors=neighbors_pt, name='self.pt0')
        
        self.down_sample1 = down_sample(dim=16, neighbors=self.neighbors_sample, sampling_rate=self.sampling_rate, name='self.down_sample1')
        self.pt_down1 = point_transformer(dim=16, neighbors=neighbors_pt, name='self.pt_down1')

        self.down_sample2 = down_sample(dim=32, neighbors=self.neighbors_sample, sampling_rate=self.sampling_rate, name='self.down_sample2') 
        self.pt_down2 = point_transformer(dim=32, neighbors=neighbors_pt, name='self.pt_down2')

        self.down_sample3 = down_sample(dim=64, neighbors=self.neighbors_sample, sampling_rate=self.sampling_rate, name='self.down_sample3')
        self.pt_down3 = point_transformer(dim=64, neighbors=neighbors_pt, name='self.pt_down3')

        self.up_sample1 = up_sample(dim=32, neighbors=self.neighbors_sample, name='self.up_sample1')
        self.pt_up1 = point_transformer(dim=32, neighbors=neighbors_pt, name='self.pt_up1')

        self.up_sample2 = up_sample(dim=16, neighbors=self.neighbors_sample, name='self.up_sample2')
        self.pt_up2 = point_transformer(dim=16, neighbors=neighbors_pt, name='self.pt_up2')

        self.up_sample3 = up_sample(dim=8, neighbors=self.neighbors_sample, name='self.up_sample3')
        self.pt_up3 = point_transformer(dim=8, neighbors=neighbors_pt, name='self.pt_up3')
            
        self.out = self.call(self.input_layer)
        
        super(point_transformer_network, self).__init__(inputs=self.input_layer, outputs=self.out, **kwargs)

    def build(self):
        # Initialize the graph
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.out
        )

    def call(self, points, training=False):
        
        pos = points[:, :, 0:3]
        feat = points[:, :, 3:]
       
        out1 = self.MLP1(feat)
        out_down1, pos_down1 = self.down_sample1(out1, pos)
        out_pt_down1 = self.pt_down1(out_down1, pos_down1)
        
        out_down2, pos_down2 = self.down_sample2(out_pt_down1, pos_down1)
        out_pt_down2 = self.pt_down2(out_down2, pos_down2)
        out_down3, pos_down3 = self.down_sample3(out_pt_down2, pos_down2)
        out_pt_down3 = self.pt_down3(out_down3, pos_down3)
        out2 = self.MLP2(out_pt_down3)
        
        out_up1 = self.up_sample1(out2, pos_down3, pos_down2)
        out_up1 = tf.concat((out_up1, out_pt_down2), axis=-1)
        out_pt_up1 = self.pt_up1(out_up1, pos_down2)

        out_up2 = self.up_sample2(out_pt_up1, pos_down2, pos_down1)
        out_up2 = tf.concat((out_up2, out_pt_down1), axis=-1)
        out_pt_up2 = self.pt_up2(out_up2, pos_down1)

        out_up3 = self.up_sample3(out_pt_up2, pos_down1, pos)
        out_up3 = tf.concat((out_up3, out1), axis=-1)
        out_pt_up3 = self.pt_up3(out_up3, pos)
        
        out_final = self.MLP3(out_pt_up3)
        out_final_norm, _ = tf.linalg.normalize(out_final, axis=-1)
        out_final_scaled = out_final_norm * tf.constant(10, dtype=tf.float32)
        
        return out_final_scaled
  



