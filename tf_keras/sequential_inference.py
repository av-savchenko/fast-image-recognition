import sys
import os.path
import os
import math
import numpy as np
import re
import glob
import argparse
import warnings
import time,datetime
from random import shuffle,seed

import _pickle as pickle

from sklearn.metrics import recall_score,accuracy_score, precision_recall_curve
from sklearn import preprocessing, model_selection
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsOneClassifier
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline


import numpy as np
RANDOM_SEED=123
np.random.seed(RANDOM_SEED)  # for reproducibility
seed(RANDOM_SEED)

VALID_IMAGE_FORMATS = frozenset(['jpg', 'jpeg'])
#IMG_SIZE=299
IMG_SIZE=224
BATCH_SIZE=8
finetune=False


from tensorflow.python.platform import gfile
from keras.applications import vgg19, inception_v3, resnet50,inception_resnet_v2, resnet_v2, mobilenet,mobilenetv2
from keras.layers import Flatten, Dense, Dropout,GlobalAveragePooling2D,AveragePooling2D, Activation, Conv2D, Lambda, Input, Reshape
from keras.models import Model,load_model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
import keras.applications
from keras.preprocessing.image import (ImageDataGenerator, Iterator,
                                       array_to_img, img_to_array, load_img)
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import efficientnet.keras as enet


class CustomImageDataGenerator(ImageDataGenerator):
    def flow_from_image_lists(self, image_lists,
                              category, image_dir,
                              target_size=(256, 256), color_mode='rgb',
                              class_mode='categorical',
                              batch_size=32, shuffle=True, seed=None,
                              save_to_dir=None,
                              save_prefix='',
                              save_format='jpeg'):
        return ImageListIterator(
            image_lists, self,
            category, image_dir,
            target_size=target_size, color_mode=color_mode,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class ImageListIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        image_lists: Dictionary of training images for each label.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, image_lists, image_data_generator,
                 category, image_dir,
                 target_size=(256, 256), color_mode='rgb',
                 class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if data_format is None:
            data_format = K.image_data_format()

        classes = list(image_lists.keys())
        self.category = category
        self.num_classes = len(classes)
        self.image_lists = image_lists
        self.image_dir = image_dir

        how_many_files = 0
        for label_name in classes:
            for _ in self.image_lists[label_name][category]:
                how_many_files += 1

        self.samples = how_many_files
        self.class2id = dict(zip(classes, range(len(classes))))
        self.id2class = dict((v, k) for k, v in self.class2id.items())
        self.classes = np.zeros((self.samples,), dtype='int32')

        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        i = 0
        self.filenames = []
        for label_name in classes:
            for j, _ in enumerate(self.image_lists[label_name][category]):
                self.classes[i] = self.class2id[label_name]
                img_path = get_image_path(self.image_lists,
                                          label_name,
                                          j,
                                          self.image_dir,
                                          self.category)
                self.filenames.append(img_path)
                i += 1

        print("Found {} {} files".format(len(self.filenames), category))
        super(ImageListIterator, self).__init__(self.samples, batch_size, shuffle,
                                                seed)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
        
    def _get_batches_of_transformed_samples(self, index_array):
        current_batch_size=len(index_array)
        batch_x = np.zeros((current_batch_size,) + self.image_shape,
                           dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            img = load_img(self.filenames[j],
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(10000),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes),
                               dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

def create_image_lists(image_dir, train_percent=0.5,train_count=-1):
    if not os.path.isdir(image_dir):
        raise ValueError("Image directory {} not found.".format(image_dir))
    image_lists = {}
    sub_dirs = [x[0] for x in os.walk(image_dir)]
    sub_dirs_without_root = sub_dirs[1:]  # first element is root directory
    num_classes=0
    for sub_dir in sub_dirs_without_root:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        #print("Looking for images in '{}'".format(dir_name))
        for extension in VALID_IMAGE_FORMATS:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        num_classes+=1
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        validation_images = []
        shuffle(file_list)
        if train_count<0:
            train_count=int(math.ceil(train_percent*len(file_list)))
        #print(label_name,train_percent,len(file_list),train_count)
        for i,file_name in enumerate(file_list):
            base_name = os.path.basename(file_name)
            if i < train_count:
                training_images.append(base_name)
            #elif i<train_count+15:
            else:
                validation_images.append(base_name)
        image_lists[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'validation': validation_images,
        }
    return image_lists,num_classes


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def get_image_path(image_lists, label_name, index, image_dir, category):
    """"Returns a path to an image for a label at the given index.

    # Arguments
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Int offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of set to pull images from - training, testing, or
      validation.

    # Returns
      File system path string to an image that meets the requested parameters.
    """
    if label_name not in image_lists:
        raise ValueError('Label does not exist ', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        raise ValueError('Category does not exist ', category)
    category_list = label_lists[category]
    if not category_list:
        raise ValueError('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_generators(image_lists, image_dir,preprocess_fct,batch_size):
    train_datagen = CustomImageDataGenerator(preprocessing_function=preprocess_fct)
    test_datagen = CustomImageDataGenerator(preprocessing_function=preprocess_fct)

    print('IMG SIZE:',IMG_SIZE)
    train_generator = train_datagen.flow_from_image_lists(
        image_lists=image_lists,
        category='training',
        image_dir=image_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',seed=RANDOM_SEED)

    validation_generator = test_datagen.flow_from_image_lists(
        image_lists=image_lists,
        category='validation',
        image_dir=image_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',seed=RANDOM_SEED)

    return train_generator, validation_generator

def create_function(input,outputs):
    func=K.function([input], outputs)
    if False:
        print(input, outputs)
        print(func)
        print(input.shape)
        tmp_x=np.expand_dims(np.ones(input.shape[1:]),axis=0)
        print(tmp_x.shape)
        tmp_y=func([tmp_x])
        #print(tmp_y)
        for i,y in enumerate(tmp_y):
            print(i,y.shape)
    return func
    
def get_model(model_file='',classes_num=None):    
    global IMG_SIZE
    #config=tf.ConfigProto(device_count = {'GPU' : 0})
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    middle_layers=[]
    net_model=None
    num_layers_to_drop=0
    if True:
        basemodel = enet
        net_model=enet.EfficientNetB7(weights=None)
        net_model.load_weights('enet_pretrained/efficientnet-b7-weights.h5') #-train
        num_layers_to_drop=2
        IMG_SIZE = net_model.input_shape[1]
        #middle_layers=['block4a_expand_conv','block5a_expand_conv','block6a_expand_conv','block7a_expand_conv']
        middle_layers=['block5b_add','block5f_add','block6b_add','block6f_add','block7b_add']
        #middle_layers=['block6b_add'] #['block6a_expand_conv'] #
    elif False:
        basemodel = vgg19
        basemodel_class = basemodel.VGG19
        num_layers_to_drop=1
    elif False:
        basemodel = inception_v3
        basemodel_class = basemodel.InceptionV3
    elif False:
        basemodel=mobilenet
        basemodel_class = basemodel.MobileNet
        middle_layers=['block_14_add']
    elif False:
        basemodel=mobilenetv2
        basemodel_class = basemodel.MobileNetV2
        middle_layers=['block_14_add']
    elif False:
        basemodel=inception_resnet_v2
        basemodel_class = basemodel.InceptionResNetV2
        middle_layers=['mixed_5b','block17_16_ac','mixed_7a','block8_5_ac']
        #middle_layers=['mixed_5b','block8_5_ac'] #['block17_17_ac','block8_5_ac'] # ['block17_16_ac','mixed_7a','block8_10']# ['block17_19_ac'] #['mixed_7a'] #
    elif True:
        basemodel = resnet_v2
        basemodel_class = basemodel.ResNet152V2
        middle_layers=['conv4_block1_out','conv4_block18_out','conv4_block36_out']
        #middle_layers=['conv4_block1_out'] #['conv4_block36_out'] #['conv5_block1_out'] #
    else:
        basemodel = resnet50
        basemodel_class = basemodel.ResNet50
        middle_layers=['activation_22', 'activation_40']
    
    if num_layers_to_drop>0:
        if net_model is None:
            net_model = basemodel_class(weights='imagenet',include_top=True, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        for _ in range(num_layers_to_drop):
            net_model.layers.pop()
        last_model_layer = net_model.layers[-1].output
    else:
        #print(model_file,os.path.exists(model_file))
        if os.path.exists(model_file):
            pass #net_model=load_model(model_file)
        if net_model is None:
            if basemodel==mobilenetv2:
                net_model = basemodel_class(alpha=1.4,weights='imagenet',include_top=False,input_shape=(IMG_SIZE, IMG_SIZE, 3),pooling='avg')
            else:
                net_model = basemodel_class(weights='imagenet',include_top=False,input_shape=(IMG_SIZE, IMG_SIZE, 3),pooling='avg')
        last_model_layer = net_model.output

    net_model.trainable=False
    for layer in net_model.layers:
        layer.trainable=False

    print(basemodel)
    #net_model.summary()
    #sys.exit(0)
    functions=[]
    embedding_sizes=[]
    input=net_model.input
    for layer_name in middle_layers:
        layer=net_model.get_layer(layer_name)
        avg_pool_layer=GlobalAveragePooling2D()(layer.output)
        l2_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(avg_pool_layer)
        l2_norm = avg_pool_layer
        
        func=create_function(input, [layer.output,l2_norm])
        functions.append(func)
        embedding_sizes.append(l2_norm.shape[1])
        input=layer.output
    
    if True:
        #preds = Lambda(lambda  x: K.l2_normalize(x,axis=1))(last_model_layer)
        preds = last_model_layer
        func=create_function(input, [last_model_layer,preds])
        functions.append(func)
        embedding_sizes.append(preds.shape[1])
    return functions,embedding_sizes,basemodel.preprocess_input, net_model,middle_layers
    
def classifier_tester(classifier,x_train,y_train,x_val,y_val, calibrated=False):
    start_time=time.time()
    if not calibrated:
        classifier.fit(x_train,y_train)
    else:
        X_tr, X_val, Y_tr, Y_val = model_selection.train_test_split(x_train,y_train, test_size=0.5, random_state=42, stratify=y_train)
        classifier.fit(X_tr, Y_tr)
        new_classifier=CalibratedClassifierCV(classifier, cv='prefit')
        new_classifier.fit(X_val,Y_val)
        #classifier.fit(x_train,y_train)
        classifier=new_classifier
    elapsed_time = time.time() - start_time
    print('classifier training elapsed=',elapsed_time)

    start_time=time.time()
    #pred_labels=classifier.predict(x_val)
    pred_labels=np.array([classifier.predict(x.reshape(1,-1))[0] for x in x_val])
    elapsed_time = time.time() - start_time
    
    macro_recall=recall_score(y_val,pred_labels,average='macro')
    elapsed_per_image=1000*elapsed_time/y_val.shape[0]
    print('recall (macro):',macro_recall,'recall (micro):',recall_score(y_val,pred_labels,average='micro'),'elapsed per image:',elapsed_per_image,' ms') #,'acc:',accuracy_score(y_val,pred_labels)
    return macro_recall,elapsed_per_image
    
from sklearn.metrics.pairwise import euclidean_distances
def knn_tester(x_train,y_train,x_val,y_val):
    start_time=time.time()
    num_tests=y_val.shape[0]
    pred_labels=np.zeros(num_tests)
    for i in range(num_tests):
        #distances=np.linalg.norm(x_train-x_val[i],axis=1)
        #distances=euclidean_distances(x_train,x_val[i].reshape(1,-1),squared=True).reshape(-1)
        distances=2-2*np.dot(x_train,x_val[i])
        #indices=distances.argsort()
        #min_ind=indices[0]
        min_ind=distances.argmin()
        
        pred_labels[i]=y_train[min_ind]
        
    elapsed_time = time.time() - start_time
    
    macro_recall=recall_score(y_val,pred_labels,average='macro')
    elapsed_per_image=1000*elapsed_time/y_val.shape[0]
    print('recall (macro):',macro_recall,'recall (micro):',recall_score(y_val,pred_labels,average='micro'),'elapsed per image:',elapsed_per_image,' ms') #,'acc:',accuracy_score(y_val,pred_labels)
    return macro_recall,elapsed_per_image

def sequential_knn_tester(x_train_all,y_train,x_val_all,y_val, check_len=True):
    num_levels=len(x_train_all)
    num_tests=y_val.shape[0]
    pred_labels=np.zeros(num_tests)
    break_counts=np.zeros(num_levels)
    #num_levels=2
    
    start_time=time.time()
    for i in range(num_tests):
        for l in range(num_levels):
            distances=2-2*np.dot(x_train_all[l],x_val_all[l][i])
            min_ind=distances.argmin()
            y_best=y_train[min_ind]
            threshold=distances[min_ind]/0.8
            if (np.all(y_train[distances<=threshold]==y_best) or l==num_levels-1):
                pred_labels[i]=y_best
                break_counts[l]+=1
                break
            
    elapsed_time = time.time() - start_time
    print('average breaks per layer:',break_counts/num_tests)

    macro_recall=recall_score(y_val,pred_labels,average='macro')
    elapsed_per_image=1000*elapsed_time/y_val.shape[0]
    print('recall (macro):',macro_recall,'recall (micro):',recall_score(y_val,pred_labels,average='micro'),'elapsed per image:',elapsed_per_image,' ms') #,'acc:',accuracy_score(y_val,pred_labels)
    return macro_recall,elapsed_per_image,break_counts/num_tests

def sequential_knn_tester_orig(x_train_all,y_train,x_val_all,y_val):
    num_levels=len(x_train_all)
    num_tests=y_val.shape[0]
    pred_labels=np.zeros(num_tests)
    break_counts=np.zeros(num_levels)
    #num_levels=2
    
    start_time=time.time()
    for i in range(num_tests):
        #indices=np.array(range(y_train.shape[0]))
        for l in range(num_levels):
            #distances=np.linalg.norm(x_train-x_val[i],axis=1)
            #print(np.linalg.norm(x_train_all[l],axis=1))
            if False:
                #tr=x_train_all[l]
                #if l>0:
                #    tr=tr[indices]
                #a=x_train_all[l]
                #cols=np.array(range(a.shape[1]))
                #tr=(a.ravel()[(cols+(indices*a.shape[1]).reshape((-1,1))).ravel()]).reshape(indices.size,cols.size)
                
                #distances=euclidean_distances(tr,x_val_all[l][i].reshape(1,-1),squared=True).reshape(-1)
                #distances=2-2*np.dot(tr,x_val_all[l][i])
                #distances=np.array([2-2*np.dot(x_train_all[l][train_ind],x_val_all[l][i]) for train_ind in indices])
                distances=2-2*np.dot(x_train_all[l],x_val_all[l][i])[indices]
                
                min_ind=distances.argmin()
                min_distance=distances[min_ind]

                new_indices=indices[(min_distance/distances)>=0.8]
                #print(l,min_distance,len(indices),indices.shape,new_indices.shape,distances.shape)
                if len(np.unique(y_train[new_indices]))==1 or l==num_levels-1:
                    pred_labels[i]=y_train[indices[min_ind]]
                    break
                indices=new_indices
            elif True:
                distances=2-2*np.dot(x_train_all[l],x_val_all[l][i])
                min_ind=distances.argmin()
                y_best=y_train[min_ind]
                terminate=False
                if check_len:
                    min_distance=distances[min_ind]
                    terminate=True #(l==num_levels-1) #(len(np.unique(y_train[(min_distance/distances)>=0.8]))==1 or l==num_levels-1)
                else:
                    threshold=distances[min_ind]/0.8
                    terminate=(np.all(y_train[distances<=threshold]==y_best) or l==num_levels-1)
                #if np.all(y_train[distances<=threshold]==y_best) or l==num_levels-1:
                if terminate:
                    pred_labels[i]=y_best
                    break_counts[l]+=1
                    break
            else:
                #distances=euclidean_distances(x_train_all[l],x_test,squared=True).reshape(-1)
                distances=2-2*np.dot(x_train_all[l],x_val_all[l][i])
                part_indices=np.argsort(distances)
                #print(part_indices)
                min_ind=part_indices[0]
                
                sorted_distances=distances[part_indices]
                y_sorted=y_train[part_indices]
                second_ind=np.argmax(y_sorted>y_sorted[0])
                dist_ratio=distances[min_ind]/sorted_distances[second_ind]
                #print(i,l,min_ind,second_ind,dist_ratio,distances[min_ind],sorted_distances[0],sorted_distances[second_ind],y_sorted[:second_ind],sorted_distances[:second_ind])
                if dist_ratio<=0.5 or l==num_levels-1:
                    pred_labels[i]=y_sorted[0]
                    break
            
    elapsed_time = time.time() - start_time
    print('average breaks per layer:',break_counts/num_tests)

    macro_recall=recall_score(y_val,pred_labels,average='macro')
    elapsed_per_image=1000*elapsed_time/y_val.shape[0]
    print('recall (macro):',macro_recall,'recall (micro):',recall_score(y_val,pred_labels,average='micro'),'elapsed per image:',elapsed_per_image,' ms') #,'acc:',accuracy_score(y_val,pred_labels)
    return macro_recall,elapsed_per_image,break_counts/num_tests

All_LAYERS=False
#from copy import deepcopy
def sequential_classifier_tester(x_train_all,y_train,x_val_all,y_val):
    num_levels=len(x_train_all)
    #num_levels=2
    if All_LAYERS:
        levels=list(range(num_levels))
    else:
        levels=[num_levels-3,num_levels-2,num_levels-1]
    
    num_tests=y_val.shape[0]

    classifiers=[]
    thresholds=[]
    #num_levels-=1
    start_time=time.time()
    for l in levels:
        svc=LinearSVC()
        #if l<num_levels-1:
        #    X_tr, X_val, Y_tr, Y_val = model_selection.train_test_split(x_train_all[l],y_train, test_size=0.5, random_state=42, stratify=y_train)
        #    svc.fit(X_tr, Y_tr)
        #    new_classifier=CalibratedClassifierCV(svc, cv='prefit')
        #    new_classifier.fit(X_val,Y_val)

        best_threshold=-1
        if l<num_levels-1:
            X_tr, X_val, Y_tr, Y_val = model_selection.train_test_split(x_train_all[l],y_train, test_size=0.5, random_state=42, stratify=y_train)
            svc.fit(X_tr, Y_tr)
            
            decision_values = svc.decision_function(X_val)
            predictions=decision_values.argmax(axis=1)
            max_decision_values=decision_values.max(axis=1)
            mistakes=max_decision_values[predictions!=Y_val]
            #for i,threshold in enumerate(sorted(max_decision_values[predictions==Y_val])[::-1]):
            #    tpr=i/len(predictions)
            #    fpr=(mistakes>threshold).sum()/len(predictions)
            #    print(threshold, tpr, fpr)
            for i,threshold in enumerate(sorted(max_decision_values[predictions==Y_val])[::-1]):
                tpr=i/len(predictions)
                fpr=(mistakes>threshold).sum()/len(predictions)
                if fpr>0.01:
                    if best_threshold==-1:
                        best_threshold=threshold
                    print('best_threshold',best_threshold)
                    break
                best_threshold=threshold
        thresholds.append(best_threshold)
        
        svc.fit(x_train_all[l],y_train)
        classifiers.append(svc)
        #print('coefs:',svc.coef_.shape, svc.intercept_.shape )

    elapsed_time = time.time() - start_time
    print('classifier training elapsed=',elapsed_time)
    
    pred_labels=np.zeros(num_tests)    
    break_counts_fixed=np.zeros(num_levels)
    start_time=time.time()
    for i in range(num_tests):
        #indices=np.array(range(y_train.shape[0]))
        for l,cls in zip(levels,classifiers):
            #pred_scores=classifiers[l].decision_function(x_val_all[l][i].reshape(1,-1)).reshape(-1)
            pred_scores=np.dot(cls.coef_,x_val_all[l][i]) + cls.intercept_
            #pred_scores=np.array([classifiers[l].predict_proba(x.reshape(1,-1))[0] for x in x_val_all[l]])
            #indices = np.argpartition(pred_scores, -2)[-2:]
            #indices = indices[np.argsort(-pred_scores[indices])]
            #best_ind=indices[0]
            #max_score=pred_scores[best_ind]
            #print(indices,best_ind,max_score)#, pred_scores)
            max_score=pred_scores.max()
            if l==num_levels-1 or max_score>0.06:
            #if pred_scores[indices[1]]<0.075:
                pred_labels[i]=pred_scores.argmax() #best_ind
                break_counts_fixed[l]+=1
                break            
    elapsed_time = time.time() - start_time
    print('Fixed threshold: average breaks per layer:',break_counts_fixed/num_tests)

    macro_recall_fixed=recall_score(y_val,pred_labels,average='macro')
    elapsed_per_image_fixed=1000*elapsed_time/y_val.shape[0]
    print('recall (macro):',macro_recall_fixed,'recall (micro):',recall_score(y_val,pred_labels,average='micro'),'elapsed per image:',elapsed_per_image_fixed,' ms')


    pred_labels=np.zeros(num_tests)    
    break_counts=np.zeros(num_levels)
    start_time=time.time()
    for i in range(num_tests):
        for l,cls,t in zip(levels,classifiers,thresholds):
            pred_scores=np.dot(cls.coef_,x_val_all[l][i]) + cls.intercept_
            max_score=pred_scores.max()
            if l==num_levels-1 or max_score>t:
                pred_labels[i]=pred_scores.argmax()
                break_counts[l]+=1
                break            
    elapsed_time = time.time() - start_time
    print('average breaks per layer:',break_counts/num_tests)

    macro_recall=recall_score(y_val,pred_labels,average='macro')
    elapsed_per_image=1000*elapsed_time/y_val.shape[0]
    print('recall (macro):',macro_recall,'recall (micro):',recall_score(y_val,pred_labels,average='micro'),'elapsed per image:',elapsed_per_image,' ms') #,'acc:',accuracy_score(y_val,pred_labels)

    return macro_recall,elapsed_per_image,break_counts/num_tests,macro_recall_fixed,elapsed_per_image_fixed,break_counts_fixed/num_tests

def conditional_classifier_tester(x_train_all,y_train,x_val_all,y_val):
    num_levels=len(x_train_all)
    #num_levels=2
    num_tests=y_val.shape[0]

    classifiers=[]
    start_time=time.time()
    for l in range(num_levels):
        svc=LinearSVC()
        svc.fit(x_train_all[l],y_train)
        classifiers.append(svc)

    elapsed_time = time.time() - start_time
    print('classifier training elapsed=',elapsed_time)
    
    for threshold in np.linspace(-0.7,1.2,21):
        pred_labels=np.zeros(num_tests)    
        break_counts=np.zeros(num_levels)
        start_time=time.time()
        for i in range(num_tests):
            for l in range(num_levels):
                pred_scores=np.dot(classifiers[l].coef_,x_val_all[l][i]) + classifiers[l].intercept_
                max_score=pred_scores.max()
                if l==num_levels-1 or max_score>threshold:
                    pred_labels[i]=pred_scores.argmax() #best_ind
                    break_counts[l]+=1
                    break
                
        elapsed_time = time.time() - start_time
        print('threshold:',threshold,' average breaks per layer:',break_counts/num_tests)

        macro_recall=recall_score(y_val,pred_labels,average='macro')
        elapsed_per_image=1000*elapsed_time/y_val.shape[0]
        print('recall (macro):',macro_recall,'recall (micro):',recall_score(y_val,pred_labels,average='micro'),'elapsed per image:',elapsed_per_image,' ms') #,'acc:',accuracy_score(y_val,pred_labels)
    return macro_recall,elapsed_per_image,break_counts/num_tests


def sequential_knn_classifier_tester(x_train_all,y_train,x_val_all,y_val, extract_pca=False):
    num_levels=len(x_train_all)
    num_tests=y_val.shape[0]
    pred_labels=np.zeros(num_tests)
    break_counts=np.zeros(num_levels)
    #num_levels=2
    
    if extract_pca:
        pcas=[]
        x_train_pca_all=[]
        for l in range(num_levels-1):
            pcas.append(PCA(n_components=128))
            pcas[l].fit(x_train_all[l])
            x_train_pca_all.append(pcas[l].transform(x_train_all[l]))
        
    start_time=time.time()
    classifier=LinearSVC()
    classifier.fit(x_train_all[num_levels-1],y_train)
    elapsed_time = time.time() - start_time
    print('classifier training elapsed=',elapsed_time)
    
    start_time=time.time()
    for i in range(num_tests):
        for l in range(num_levels-1):
            
            if extract_pca:
                val_pca=pcas[l].transform(x_val_all[l][i].reshape(1,-1))
                distances=euclidean_distances(x_train_pca_all[l],val_pca,squared=True).reshape(-1)
            else:
                distances=2-2*np.dot(x_train_all[l],x_val_all[l][i])
            min_ind=distances.argmin()
            threshold=distances[min_ind]/0.8
            y_best=y_train[min_ind]
            #if len(np.unique(y_train[distances>=threshold]))==1:
            if np.all(y_train[distances<=threshold]==y_best):
                pred_labels[i]=y_best
                break_counts[l]+=1
                break
            elif l==num_levels-2:
                pred_labels[i]=classifier.predict(x_val_all[num_levels-1][i].reshape(1,-1))[0]
                break_counts[l+1]+=1
            
    elapsed_time = time.time() - start_time
    print('average breaks per layer:',break_counts/num_tests)

    macro_recall=recall_score(y_val,pred_labels,average='macro')
    elapsed_per_image=1000*elapsed_time/y_val.shape[0]
    print('recall (macro):',macro_recall,'recall (micro):',recall_score(y_val,pred_labels,average='micro'),'elapsed per image:',elapsed_per_image,' ms') #,'acc:',accuracy_score(y_val,pred_labels)
    return macro_recall,elapsed_per_image,break_counts/num_tests

def chi2dist(x, y):
    sum=x+y
    chi2array=np.where(sum>0, (x-y)**2/sum, 0)
    return np.sum(chi2array)

def run_inference(functions, embedding_sizes,model_file, generators):
    train_generator, validation_generator = generators
    nb_train_samples=train_generator.samples
    nb_val_samples=validation_generator.samples
    #print('nb_train_samples',nb_train_samples)
    #print('nb_val_samples',nb_val_samples)
    num_levels=len(embedding_sizes)
    
    #network_name='places_inceptionresnet_layers_block17_17_ac_block8_5_ac_'
    #network_name='dogs_inceptionresnet_layers_block17_17_ac_block8_5_ac_'
    #network_name='dogs_inceptionresnet_layers_block17_16_ac_mixed_7a_block8_10_'
    #network_name='dogs_inceptionresnet_layers_block17_19_ac_last_'
    #network_name='dogs_inceptionresnet_layers_mixed_7a_last_'
    #network_name='dogs_resnet_layers_conv5_block1_out_last_'
    #network_name='dogs_inceptionresnet_multi_heads_ft_'
    
    #network_name='dogs_mobilenet2_1.0_layers_block_14_add_last_' #_prunned
    
    #network_name='caltech256_mobilenet_v2_ft_'
    #network_name='caltech256_inceptionresnet_ft_60_'
    #network_name='caltech101_resnet_ft_30_'
    #network_name='caltech256_resnet_ft_60_'
    
    #network_name='caltech101_mobilenet2_layers_block_14_add_last_'
    #network_name='caltech256_inceptionresnet_layers_block17_16_ac_mixed_7a_block8_10_'
    #network_name='caltech101_inceptionresnet_layers_block17_17_ac_block8_5_ac_new_'
    #network_name='caltech101_inceptionresnet_layers_block17_19_ac_last_'
    #network_name='caltech101_resnet152_layers_conv4_block36_out_last_'
    #network_name='caltech101_inceptionresnet_multi_heads_ft_'
    
    #network_name='caltech256_enet5_block6b_add_last_'
    #network_name='caltech101_enet5_block6a_expand_conv_last_'
    #network_name='dogs_enet5_block6b_add_last_'
    
    #network_name='caltech101_enet7_block6b_add_last_'
    #network_name='caltech256_enet7_all_'
    network_name='dogs_enet7_all_'

    train_file=network_name+'train.pkl'
    if os.path.exists(train_file):
        with open(train_file, 'rb') as infile:
            x_train_all,y_train,inference_times_train = pickle.load(infile)
    else:
        y_train=np.ndarray(shape=(nb_train_samples,))
        x_train_all=[]
        for ind in range(num_levels):
            x_train_all.append(np.ndarray(shape=(nb_train_samples,embedding_sizes[ind])))
        
        level_times=np.zeros(num_levels)
        start_time=time.time()
        for i in range(nb_train_samples):
            batch_x,batch_y=train_generator.next()
            input=batch_x
            for ind in range(num_levels):
                #print(ind,input.shape)
                start=time.time()
                next_input,preds=functions[ind]([input])
                level_times[ind]+=time.time() - start 
                x_train_all[ind][i]=preds[0]
                #print(next_input.shape,preds.shape)
                input=next_input
                #break
            y_train[i]=np.argmax(batch_y[0],axis=-1)
        elapsed_time = time.time() - start_time    
        
        print('training elapsed=',elapsed_time,'sec., per image:',1000*elapsed_time/nb_train_samples,' ms')
        inference_times_train=[]
        for ind in range(num_levels):
            inference_time_train=1000*level_times[ind]/nb_train_samples
            inference_times_train.append(inference_time_train)
            print('training ',(ind+1),' per image:',inference_time_train,' ms')
        with open(train_file, 'wb') as outfile:
            pickle.dump((x_train_all,y_train,inference_times_train), outfile)
    
    val_file=network_name+'val.pkl'
    if os.path.exists(val_file):
        with open(val_file, 'rb') as infile:
            x_val_all,y_val,inference_times_val = pickle.load(infile)
    else:
        y_val=np.ndarray(shape=(nb_val_samples,))
        x_val_all=[]
        for ind in range(num_levels):
            x_val_all.append(np.ndarray(shape=(nb_val_samples,embedding_sizes[ind])))
        level_times=np.zeros(num_levels)
        start_time=time.time()
        for i in range(nb_val_samples):
            batch_x,batch_y=validation_generator.next()
            input=batch_x
            for ind in range(num_levels):
                start=time.time()
                next_input,preds=functions[ind]([input])
                level_times[ind]+=time.time() - start 
                x_val_all[ind][i]=preds[0]
                #print(next_input.shape,preds.shape)
                input=next_input
                #break
            y_val[i]=np.argmax(batch_y[0],axis=-1)
        elapsed_time = time.time() - start_time    
        print('validation elapsed=',elapsed_time,'sec., per image:',1000*elapsed_time/nb_val_samples,' ms')
        
        inference_times_val=[]
        for ind in range(num_levels):
            inference_time_val=1000*level_times[ind]/nb_val_samples
            inference_times_val.append(inference_time_val)
            print('validation ',(ind+1),' per image:',inference_time_val,' ms')
        with open(val_file, 'wb') as outfile:
            pickle.dump((x_val_all,y_val,inference_times_val), outfile)
    
    inference_times=[]
    for ind in range(num_levels):
        avg_time=(inference_times_train[ind]+inference_times_val[ind])/2
        print(ind,inference_times_train[ind],inference_times_val[ind])
        if ind==0:
            inference_times.append(avg_time)
        else:
            inference_times.append(inference_times[ind-1]+avg_time)
    #print('inference_times:',inference_times)
    #return

    classifiers=[]
    #classifiers.append(['OS-PNN+PCA',Pipeline(steps=[('pca', PCA(n_components=256)), ('classifier', OSPNNClassifier())])])
    #classifiers.append(['OS-PNN',OSPNNClassifier()])
    
    #classifiers.append(['PNN+PCA',Pipeline(steps=[('pca', PCA(n_components=256)), ('classifier', PNNClassifier(bandwidth=5e-6))])])
    #classifiers.append(['PNN',PNNClassifier(bandwidth=5e-6)])
    
    #classifiers.append(['RF+PCA',Pipeline(steps=[('pca', PCA(n_components=256)), ('classifier', RandomForestClassifier(n_estimators=1000,max_depth=3))])])
    #classifiers.append(['RF',RandomForestClassifier(n_estimators=100)]) #(n_estimators=1000,max_depth=3)])
    
    #classifiers.append(['3NN+PCA',Pipeline(steps=[('pca', PCA(n_components=256)), ('classifier', KNeighborsClassifier(n_neighbors=3))])])
    #classifiers.append(['3NN',KNeighborsClassifier(n_neighbors=3)])
    
    #classifiers.append(['LinearSVM+PCA',Pipeline(steps=[('pca', PCA(n_components=256)), ('classifier', LinearSVC())])])
    #classifiers.append(['LinearSVM',LinearSVC()]) #
    #classifiers.append(['SVM (linear)',SVC(kernel='linear')])
    #classifiers.append(['SVM RBF',SVC(gamma='auto')])
    #classifiers.append(['linear one-vs-one SVM probab',OneVsOneClassifier(LinearSVC())])
    #classifiers.append(['SGDClassifier (a=best)',SGDClassifier(alpha=1.0/y_train.shape[0], loss='hinge',tol=0.0001)])
    #classifiers.append(['SGDClassifier (a=1)',SGDClassifier(alpha=1, loss='squared_hinge',tol=0.0001,max_iter=100,fit_intercept=False)])
    
    #classifiers.append(['1NN+PCA',Pipeline(steps=[('pca', PCA(n_components=256)), ('classifier', KNeighborsClassifier(n_neighbors=1))])])
    #classifiers.append(['1NN',KNeighborsClassifier(n_neighbors=1,algorithm='brute')])#,metric=chi2dist)])

    if finetune: # or True:
        all_recall,all_total_time={},{}
        test_classifiers(inference_times,x_train_all,y_train,x_val_all,y_val,classifiers,all_recall,all_total_time)
    else:
        NUM_TESTS=1 #2 #3 #1
        #train_sizes=[2,5,10,15,20,25,30] #
        train_sizes=[110]
        #train_sizes=[2,5,10,20,30,40,50,60,70]#
        #train_sizes=[2,10,30,50,70,90,110,130]
        for train_size in train_sizes:
            all_recall,all_total_time={},{}
            for i in range(NUM_TESTS):
                print('\nIteration #',(i+1))
                y_all=np.concatenate((y_train,y_val))
                all_indices=np.arange(y_all.shape[0])
                #test_size=8267 #y_val.shape[0]
                #train_indices, val_indices, y_train_new, y_val_new=model_selection.train_test_split(all_indices,y_all, test_size=test_size, random_state=42, stratify=y_all, shuffle=True)
                train_indices, val_indices = [], []
                y_train_new, y_val_new=[],[]
                for lbl in np.unique(y_all):
                    tmp_y = y_all[y_all == lbl]
                    tmp_inds = all_indices[y_all == lbl]
                    shuffle(tmp_inds)
                    #current_train_size=int(len(tmp_y) * 0.1)
                    current_train_size=train_size
                    if current_train_size<=0:
                        current_train_size=1
                    elif current_train_size>len(tmp_y):
                        current_train_size=len(tmp_y)-1
                    train_indices.append(tmp_inds[: current_train_size])
                    val_indices.append(tmp_inds[current_train_size :])
                    y_train_new.append(tmp_y[: current_train_size])
                    y_val_new.append(tmp_y[current_train_size :])
                    
                y_train_new = np.concatenate(y_train_new, axis=0)
                train_indices = np.concatenate(train_indices, axis=0)
                y_val_new = np.concatenate(y_val_new, axis=0)
                val_indices = np.concatenate(val_indices, axis=0)
                x_train_all_new=[]
                x_val_all_new=[]

                print('shapes:',y_train_new.shape,y_val_new.shape)
                for l in range(num_levels):
                    x_all=np.concatenate((x_train_all[l],x_val_all[l]))
                    x_train_all_new.append(x_all[train_indices])
                    x_val_all_new.append(x_all[val_indices])

                    print(x_train_all_new[l].shape,x_val_all_new[l].shape)

                test_classifiers(inference_times,x_train_all_new,y_train_new,x_val_all_new,y_val_new,classifiers,all_recall,all_total_time)
            
            print('!!!!! avg results for ',train_size, ' train images per class:')
            for title in all_recall:
                print(title,'recall:',all_recall[title]/NUM_TESTS,' time:',all_total_time[title]/NUM_TESTS)


def test_classifiers(inference_times,x_train_all_new,y_train_new,x_val_all_new,y_val_new,classifiers,all_recall,all_total_time):
    num_levels=len(x_train_all_new)
    if All_LAYERS:
        levels=list(range(num_levels))
    else:
        levels=[num_levels-2,num_levels-1]

    if True:
        for cls_name,classifier in classifiers:
            print(cls_name)
            for i in levels:
                print('level:',i)
                macro_recall,elapsed_per_image=classifier_tester(classifier,x_train_all_new[i],y_train_new,x_val_all_new[i],y_val_new,calibrated=False)
                elapsed_per_image+=inference_times[i]
                print('time with inference:',elapsed_per_image)
                
                title='%s_%d'%(cls_name,i+1)
                if title not in all_recall:
                    all_recall[title]=all_total_time[title]=0
                all_recall[title]+=macro_recall
                all_total_time[title]+=elapsed_per_image

                if False:
                    macro_recall,elapsed_per_image=classifier_tester(classifier,x_train_all_new[i],y_train_new,x_val_all_new[i],y_val_new,calibrated=True)
                    elapsed_per_image+=inference_times[i]
                    print('time with inference:',elapsed_per_image)
                    
                    title='calibrated %s_%d'%(cls_name,i+1)
                    if title not in all_recall:
                        all_recall[title]=all_total_time[title]=0
                    all_recall[title]+=macro_recall
                    all_total_time[title]+=elapsed_per_image

    if False:    
        print('1NN')
        for i in num_levels:
            print('level:',i)
            macro_recall,elapsed_per_image=knn_tester(x_train_all_new[i],y_train_new,x_val_all_new[i],y_val_new)
            elapsed_per_image+=inference_times[i]
            print('time with inference:',elapsed_per_image)
                
            title='1NN_%d'%(i+1)
            if title not in all_recall:
                all_recall[title]=all_total_time[title]=0
            all_recall[title]+=macro_recall
            all_total_time[title]+=elapsed_per_image

    print('sequential')
    if False:
        title='1NN+last classifier'
        print(title)
        macro_recall,elapsed_per_image,break_percents=sequential_knn_classifier_tester(x_train_all_new,y_train_new,x_val_all_new,y_val_new)
        for i in range(num_levels):
            elapsed_per_image+=inference_times[i]*break_percents[i]
        print('time with inference:',elapsed_per_image)
        if title not in all_recall:
            all_recall[title]=all_total_time[title]=0
        all_recall[title]+=macro_recall
        all_total_time[title]+=elapsed_per_image
    
    
    if False:
        title='1NN only'
        print(title)
        #macro_recall,elapsed_per_image,break_percents=sequential_knn_tester(x_train_all_new,y_train_new,x_val_all_new,y_val_new, check_len=False)
        macro_recall,elapsed_per_image,break_percents=sequential_knn_tester(x_train_all_new,y_train_new,x_val_all_new,y_val_new, check_len=True)
        for i in range(num_levels):
            elapsed_per_image+=inference_times[i]*break_percents[i]
        print('time with inference:',elapsed_per_image)
        if title not in all_recall:
            all_recall[title]=all_total_time[title]=0
        all_recall[title]+=macro_recall
        all_total_time[title]+=elapsed_per_image

    if True:
        title='classifier only'
        print(title)
        #num_levels=2
        macro_recall,elapsed_per_image,break_percents,macro_recall_fixed,elapsed_per_image_fixed,break_percents_fixed=sequential_classifier_tester(x_train_all_new,y_train_new,x_val_all_new,y_val_new)
        #macro_recall,elapsed_per_image,break_percents=conditional_classifier_tester(x_train_all_new,y_train_new,x_val_all_new,y_val_new)
        
        for i in range(num_levels):
            elapsed_per_image+=inference_times[i]*break_percents[i]
        print('time with inference:',elapsed_per_image)
        if title not in all_recall:
            all_recall[title]=all_total_time[title]=0
        all_recall[title]+=macro_recall
        all_total_time[title]+=elapsed_per_image

        title='classifier only (fixed threshold)'
        for i in range(num_levels):
            elapsed_per_image_fixed+=inference_times[i]*break_percents_fixed[i]
        print('time with inference fixed:',elapsed_per_image_fixed)
        if title not in all_recall:
            all_recall[title]=all_total_time[title]=0
        all_recall[title]+=macro_recall_fixed
        all_total_time[title]+=elapsed_per_image_fixed
        

        
def run_branchynet_inference(full_model, generators):
    train_generator, validation_generator = generators
    classes_num=train_generator.num_classes
    nb_val_samples=validation_generator.samples
    
    #img=np.ones((1,224,224,3))
    #out=full_model.predict(img)
    #print(len(out),[x.shape for x in out])
    #sys.exit()
    
    #print('nb_val_samples',nb_val_samples)
    #val_file='caltech101_5_resnet_val.pkl'
    val_file='caltech256_60_resnet_val.pkl'
    if os.path.exists(val_file):
        with open(val_file, 'rb') as infile:
            preds_val_all,y_val,predict_elapsed_time = pickle.load(infile)
        print('validation time pure:', predict_elapsed_time,' ms')
        num_levels=len(preds_val_all)
    else:
        num_levels=len(full_model.outputs)
        print(full_model.outputs,num_levels)
    
        y_val=np.ndarray(shape=(nb_val_samples,))
        preds_val_all=[[] for ind in range(num_levels)]
        predict_elapsed_time=0
        start_time=time.time()
        for i in range(nb_val_samples):
            batch_x,batch_y=validation_generator.next()
            input=batch_x
            predict_start_time=time.time()
            out=full_model.predict(input)
            predict_elapsed_time += time.time() - predict_start_time
            for ind in range(num_levels):
                preds_val_all[ind].append(out[ind])
            y_val[i]=np.argmax(batch_y[0],axis=-1)
        elapsed_time = time.time() - start_time
        predict_elapsed_time=1000*predict_elapsed_time/nb_val_samples
        print('validation elapsed=',elapsed_time,'sec., per image:',1000*elapsed_time/nb_val_samples,' ms, pure:', predict_elapsed_time,' ms')

        with open(val_file, 'wb') as outfile:
            pickle.dump((preds_val_all,y_val,predict_elapsed_time), outfile)

            
    for ind in range(num_levels):
        y_preds=[np.argmax(preds,axis=-1) for preds in preds_val_all[ind]]
        macro_recall=recall_score(y_val,y_preds,average='macro')
        print('level:',ind,' recall (macro):',macro_recall)
    
    if True:
        print('BranchyNet (entropy):')
        max_entropy=math.log(classes_num)
        for threshold in np.linspace(0,max_entropy,101)[::-1]:
            y_preds=[]
            break_counts=np.zeros(num_levels)
            for i in range(nb_val_samples):
                for ind in range(num_levels):
                    entropy=-np.sum(preds_val_all[ind][i][0]*np.log(preds_val_all[ind][i][0]))
                    if entropy<=threshold or ind==num_levels-1:
                        best_ind=np.argmax(preds_val_all[ind][i])
                        break_counts[ind]+=1
                        y_preds.append(best_ind)
                        break
            break_counts/=nb_val_samples
            
            macro_recall=recall_score(y_val,y_preds,average='macro')
            print('threshold:',threshold,' recall (macro):',macro_recall,' break_counts=',break_counts)
            if break_counts[-1]>0.99:
                break
    else:
        print('ConditionalNet:')
        for threshold in np.linspace(0,1,101):
            y_preds=[]
            break_counts=np.zeros(num_levels)
            for i in range(nb_val_samples):
                for ind in range(num_levels):
                    best_ind=np.argmax(preds_val_all[ind][i])
                    #print(best_ind,preds_val_all[ind][i][0][best_ind])
                    if preds_val_all[ind][i][0][best_ind]>threshold or ind==num_levels-1:
                        break_counts[ind]+=1
                        y_preds.append(best_ind)
                        break
            break_counts/=nb_val_samples
            
            macro_recall=recall_score(y_val,y_preds,average='macro')
            print('threshold:',threshold,' recall (macro):',macro_recall,' break_counts=',break_counts)
            if break_counts[-1]>0.99:
                break
        
    sys.exit()


def train(net_model,generators,middle_layers=None):
    train_generator, val_generator = generators
    classes_num=train_generator.num_classes
    nb_train_samples=train_generator.samples
    nb_validation_samples=val_generator.samples
    print(classes_num,nb_train_samples,nb_validation_samples)
    
    #net_model.summary()
    preds=[]
    if True:
        if middle_layers is not None:
            for layer_name in middle_layers:
                layer=net_model.get_layer(layer_name)
                avg_pool_layer=GlobalAveragePooling2D()(layer.output)
                preds.append(Dense(classes_num, activation='softmax')(avg_pool_layer))
    last_model_layer = net_model.output
    #last_model_layer = GlobalAveragePooling2D()(last_model_layer)
    preds.append(Dense(classes_num, activation='softmax')(last_model_layer))
    f_model = Model(net_model.input, preds)
    if False:
        f_model.load_weights('caltech256_resnet_multi_heads_fast_60-08-5.44.h5')
        net_model.save_weights('caltech256_resnet_multi_heads_base_fast_60-08-5.44.h5')
        sys.exit(0)
        #return f_model
    elif True:
        net_model.load_weights('caltech256_resnet_multi_heads_base_fast_60-08-5.44.h5')
        #f_model.load_weights('caltech256_inceptionresnet_multi_heads_fast_60-08-11.05.h5')
        return f_model
    
    def generate_data_generator(generator):
        while True:
            batch_x, batch_y = generator.next()
            #print(batch_x.shape,batch_y.shape)
            yield batch_x, [batch_y]*len(preds)
    
    loss_weights=[len(preds)-i for i in range(len(preds))]
    #loss_weights=([1.5]*(len(preds)-1))
    #loss_weights.extend([1])
    print('loss_weights',loss_weights)

    net_model.trainable=False
    for l in net_model.layers:
        l.trainable=False
    f_model.compile(loss=['categorical_crossentropy']*len(preds), loss_weights=loss_weights,optimizer=SGD(lr=0.01), metrics=['accuracy'])
    #f_model.summary()

    FIRST_EPOCHS=2
    #mc = ModelCheckpoint('dogs_mobilenet_multi_heads-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', verbose=1, save_best_only=True)
    mc = ModelCheckpoint('dogs_resnet_multi_heads_fast_10-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=1, save_best_only=True)
    hist1=f_model.fit_generator(generate_data_generator(train_generator), steps_per_epoch=nb_train_samples//BATCH_SIZE, epochs=FIRST_EPOCHS, verbose=1, initial_epoch=0, callbacks=[mc], validation_data=generate_data_generator(val_generator), validation_steps=nb_validation_samples // BATCH_SIZE)

    net_model.trainable=True
    for l in net_model.layers:
        l.trainable=True
    f_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])
    #f_model.summary()

    SECOND_EPOCHS=FIRST_EPOCHS+6
    #es=EarlyStopping(monitor='val_acc',patience=2)
    #mc = ModelCheckpoint('model-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', verbose=1, save_best_only=True)
    hist1=f_model.fit_generator(generate_data_generator(train_generator), steps_per_epoch=nb_train_samples//BATCH_SIZE, epochs=SECOND_EPOCHS, verbose=1, 
                    initial_epoch=len(hist1.history['loss']), callbacks=[mc], validation_data=generate_data_generator(val_generator), validation_steps=nb_validation_samples // BATCH_SIZE)
    return f_model                
    
    
def test_time(largeNet=False):
    if largeNet:
        net=inception_resnet_v2.InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(224,224, 3),pooling='avg')
    else:
        net=mobilenetv2.MobileNetV2(alpha=1.4,weights='imagenet',include_top=False,input_shape=(224, 224, 3),pooling='avg')
    #img=np.random.rand(1,224,224,3)
    img=np.ones((1,224,224,3))
    out=net.predict(img)
    print(out.shape,out)
    NUM_TESTS=10
    start_time=time.time()
    for _ in range(NUM_TESTS):
        res=net.predict(img)
    
    elapsed_time = time.time() - start_time    
    print('test elapsed=',elapsed_time,'sec., per test:',1000*elapsed_time/NUM_TESTS,' ms')

def test_functions(functions):
    #img=np.random.rand(1,IMG_SIZE,IMG_SIZE,3)
    img=np.ones((1,IMG_SIZE,IMG_SIZE,3))
    NUM_TESTS=100
    
    num_levels=len(functions)
    input=img
    for ind in range(num_levels):
        start=time.time()
        next_input,preds=functions[ind]([input])
        input=next_input
    print(next_input.shape)#,next_input)

    level_times=np.zeros(num_levels)
    start_time=time.time()
    for _ in range(NUM_TESTS):
        input=img
        for ind in range(num_levels):
            start=time.time()
            next_input,preds=functions[ind]([input])
            level_times[ind]+=1000*(time.time() - start) 
            input=next_input
    elapsed_time = time.time() - start_time    
    print('test elapsed=',elapsed_time,'sec., per test:',1000*elapsed_time/NUM_TESTS,' ms', ' per level:',level_times/NUM_TESTS)


def proposed_pipeline(net_model,middle_layers, functions, embedding_sizes, generators,num_classes):
    train_generator, validation_generator = generators
    nb_train_samples=train_generator.samples
    nb_val_samples=validation_generator.samples
    num_levels=len(embedding_sizes)
    
    y_train=np.ndarray(shape=(nb_train_samples,))
    x_train_all=[]
    for ind in range(num_levels):
        x_train_all.append(np.ndarray(shape=(nb_train_samples,embedding_sizes[ind])))
    level_times=np.zeros(num_levels)
    start_time=time.time()
    for i in range(nb_train_samples):
        batch_x,batch_y=train_generator.next()
        input=batch_x
        for ind in range(num_levels):
            start=time.time()
            next_input,preds=functions[ind]([input])
            level_times[ind]+=time.time() - start 
            x_train_all[ind][i]=preds[0]
            input=next_input
        y_train[i]=np.argmax(batch_y[0],axis=-1)
    elapsed_time = time.time() - start_time    
    
    print('training elapsed=',elapsed_time,'sec., per image:',1000*elapsed_time/nb_train_samples,' ms')
    
    classifiers=[]
    thresholds=[]
    start_time=time.time()
    for l in range(num_levels):
        svc=LinearSVC()
        #if l<num_levels-1:
        #    X_tr, X_val, Y_tr, Y_val = model_selection.train_test_split(x_train_all[l],y_train, test_size=0.5, random_state=42, stratify=y_train)
        #    svc.fit(X_tr, Y_tr)
        #    new_classifier=CalibratedClassifierCV(svc, cv='prefit')
        #    new_classifier.fit(X_val,Y_val)

        if l<num_levels-1:
            X_tr, X_val, Y_tr, Y_val = model_selection.train_test_split(x_train_all[l],y_train, test_size=0.5, random_state=42, stratify=y_train)
            svc.fit(X_tr, Y_tr)
            
            decision_values = svc.decision_function(X_val)
            predictions=decision_values.argmax(axis=1)
            max_decision_values=decision_values.max(axis=1)
            mistakes=max_decision_values[predictions!=Y_val]
            #for i,threshold in enumerate(sorted(max_decision_values[predictions==Y_val])[::-1]):
            #    tpr=i/len(predictions)
            #    fpr=(mistakes>threshold).sum()/len(predictions)
            #    print(threshold, tpr, fpr)
            best_threshold=-1
            for i,threshold in enumerate(sorted(max_decision_values[predictions==Y_val])[::-1]):
                tpr=i/len(predictions)
                fpr=(mistakes>threshold).sum()/len(predictions)
                if fpr>0.01:
                    if best_threshold==-1:
                        best_threshold=threshold
                    print('best_threshold',best_threshold)
                    break
                best_threshold=threshold
            thresholds.append(best_threshold)
        
        svc.fit(x_train_all[l],y_train)
        classifiers.append(svc)
        #print('coefs:',svc.coef_.shape, svc.intercept_.shape )

    elapsed_time = time.time() - start_time
    print('classifier training elapsed=',elapsed_time)
    
    
    functions=[]
    embedding_sizes=[]
    input=net_model.input
    for l,layer_name in enumerate(middle_layers):
        layer=net_model.get_layer(layer_name)
        avg_pool_layer=GlobalAveragePooling2D()(layer.output)
        l2_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(avg_pool_layer)
        scores = Dense(num_classes,weights=[classifiers[l].coef_.T,classifiers[l].intercept_])(l2_norm)
        
        func=create_function(input, [layer.output,scores])
        functions.append(func)
        embedding_sizes.append(scores.shape[1])
        input=layer.output
            
    last_model_layer=net_model.output
    l2_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(last_model_layer)
    scores = Dense(num_classes,weights=[classifiers[-1].coef_.T,classifiers[-1].intercept_])(l2_norm)
    func=create_function(input, [last_model_layer,scores])
    functions.append(func)
    embedding_sizes.append(scores.shape[1])
    
    num_tests=0
    break_counts=np.zeros(num_levels)
    y_val=[]
    pred_labels=[]
    start_time=time.time()
    for i in range(nb_val_samples):
        batch_x,batch_y=validation_generator.next()
        y_val.append(np.argmax(batch_y[0],axis=-1))
        num_tests+=1
        input=batch_x
        for l in range(num_levels):
            next_input,preds=functions[l]([input])
            pred_scores=preds[0]
            input=next_input
            max_score=pred_scores.max()
            if l==num_levels-1 or max_score>thresholds[l]:
                pred_label=pred_scores.argmax()
                pred_labels.append(pred_label)
                break_counts[l]+=1
                break
            
    elapsed_time = time.time() - start_time

    macro_recall=recall_score(y_val,pred_labels,average='macro')
    elapsed_per_image=1000*elapsed_time/num_tests
    print('average breaks per layer:',break_counts/num_tests)
    print('recall (macro):',macro_recall,'recall (micro):',recall_score(y_val,pred_labels,average='micro'),'elapsed per image:',elapsed_per_image,' ms') #,'acc:',accuracy_score(y_val,pred_labels)

    return
    

if __name__ == '__main__':
    #test_time(False)
    #test_time(True)
    #sys.exit(0)
    model_file='' #'prunned/dogs/mobilenet_v2_1.0_l1reg_my25-pruned_short.h5' #'models/caltech101_vgg19_ft_0.3_acc77.9_weights.h5'
    #train_percent=0.3 
    train_count=60 #30 #70 #110
    #image_dir='D:/datasets/caltech/256_ObjectCategories'
    #image_dir='D:/datasets/places/all_scenes_224/val'
    image_dir='D:/datasets/StanfordDogs/dogs_all'
    
    image_lists,num_classes = create_image_lists(image_dir, train_count=train_count)
    print("Number of classes found: {}".format(num_classes))
    functions,embedding_sizes,preprocess_fct,net_model,middle_layers = get_model(model_file,num_classes)
    
    test_functions(functions)
    #sys.exit(0)
    
    if finetune and not os.path.exists(model_file):
        generators = get_generators(image_lists, image_dir,preprocess_fct, BATCH_SIZE)
        full_model=train(net_model,generators,middle_layers)
    else:
        full_model=None

    generators = get_generators(image_lists, image_dir,preprocess_fct, 1)
    if False:
        proposed_pipeline(net_model,middle_layers, functions, embedding_sizes, generators,num_classes)
    else:
        run_inference(functions, embedding_sizes,model_file, generators)
        #run_branchynet_inference(full_model, generators)


