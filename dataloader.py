import glob
import h5py
import numpy as np
import utils
from tensorflow import keras


def batch_generator_(data, label, batch_size, is_training=True):
    n = 0
    while True:
        batch_data = data[n*batch_size: (n+1)*batch_size,...]
        batch_label = label[n*batch_size: (n+1)*batch_size,...]
        if is_training==True:
            batch_data[:, :, 0:3] = utils.rotate_point_cloud(batch_data[:, :, 0:3])
        n += 1
        yield (np.asarray(batch_data), np.asarray(batch_label))
        
class batch_generator(keras.utils.Sequence):
    def __init__(self, batch_size, data, label, is_training=True, shuffle=True):
        self.batch_size = batch_size
        self.data = data
        self.label = label
        self.is_training = is_training
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(self.data.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation__(self, id_name):
        return self.data, self.label

    def __len__(self):
        return self.data.shape[0] // self.batch_size

    def __getitem__(self, index):
        batch_data = self.data[index*self.batch_size: (index+1)*self.batch_size,...]
        batch_label = self.label[index*self.batch_size: (index+1)*self.batch_size,...]
        if self.is_training==True:
            batch_data[:, :, 0:3] = utils.rotate_point_cloud(batch_data[:, :, 0:3])
        return np.asarray(batch_data), np.asarray(batch_label)  
        
        
def S3DIS_dataloader(data_path, roomlist_file, test_area_number):
    
    test_area = 'Area_' + str(test_area_number)
    ALL_FILES = sorted(glob.glob(data_path + '/*.h5'))
    room_filelist = [line.rstrip() for line in open(roomlist_file)]

    data_batch_list = []
    label_batch_list = []

    for h5_filename in ALL_FILES:
        f = h5py.File(h5_filename, 'r+')
        data_batch = f['data']
        label_batch = f['label']
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)

    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)
    
    train_idx = []
    test_idx = []

    for i, roomname in enumerate(room_filelist):
        if test_area == roomname[0:6]:
            test_idx.append(i)
        else:
            train_idx.append(i)

    train_data = data_batches[train_idx, ...]
    train_label = label_batches[train_idx, ...]
    test_data = data_batches[test_idx, ...]
    test_label = label_batches[test_idx, ...]

    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)
    
    return train_data, train_label, test_data, test_label

