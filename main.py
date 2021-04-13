import argparse
import os
import tensorflow as tf
import numpy as np

from point_transformer_network import *
from dataloader import *
from utils import *
from loss import *
import custom_metrics


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--num_points', type=int, default=4096, help='Number of points')
parser.add_argument('--save_ckpt_path', type=str, default="checkpoints/cp.ckpt", help='Directory path for saving checkpoint files')
parser.add_argument('--num_classes', type=int, default=13, help='Number of class categories (default: 13 for S3DIS dataset)')
parser.add_argument('--data_path', type=str, default='/hd/data/3d/s3dis/indoor3d_sem_seg_hdf5_data/', help='Directory path of S3DIS dataset')
parser.add_argument('--roomlist_file', type=str, default='room_filelist.txt', help='Name of roomlist file')
parser.add_argument('--test_area', type=int, default=5, help='Number of test area')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
num_points = FLAGS.num_points
num_classes = FLAGS.num_classes

checkpoint_path = FLAGS.save_ckpt_path
checkpoint_dir = os.path.dirname(checkpoint_path)


os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

def get_model(num_class=13, neighbors_sample=10, neighbors_pt = 10, sampling_rate=4, batch_size=None):

    model = point_transformer_network(num_class, neighbors_sample, neighbors_pt, sampling_rate, batch_size)
    return model

    

if __name__=='__main__':
    
    train_data, train_label, test_data, test_label = S3DIS_dataloader(data_path=FLAGS.data_path,
                                                                      roomlist_file=os.path.join(FLAGS.data_path, FLAGS.roomlist_file),
                                                                      test_area_number=FLAGS.test_area)
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='loss',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='auto')
    
    model = get_model(num_class=num_classes, batch_size=BATCH_SIZE)
    model.build()
    model_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=model_optimizer, 
                  loss=cross_entropy_with_softmax, 
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                           custom_metrics.MeanIoU_batch,
                           custom_metrics.MeanIoU_total(num_classes)])
    model.summary(line_length=120)
    
    train_result = model.fit(batch_generator(batch_size=BATCH_SIZE, data=train_data, label=train_label),
              steps_per_epoch = train_data.shape[0]//BATCH_SIZE,
              epochs=2,
              verbose=1,
              shuffle=1,
              callbacks=[checkpoint],
              validation_data=batch_generator(batch_size=BATCH_SIZE, data=test_data, label=test_label, is_training=False),
              validation_steps=test_data.shape[0]//BATCH_SIZE)

    test_result = model.evaluate(test_data, test_label, batch_size=BATCH_SIZE)
