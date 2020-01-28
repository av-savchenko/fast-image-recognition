import sys
import os.path
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import math
import numpy as np
import re
import glob
import argparse
import warnings
import time,datetime
from random import shuffle,seed

from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
  
import numpy as np
RANDOM_SEED=123
np.random.seed(RANDOM_SEED)  # for reproducibility
seed(RANDOM_SEED)

VALID_IMAGE_FORMATS = frozenset(['jpg', 'jpeg'])
IMG_SIZE=224



import sys
new_sys_dir = os.path.join('..','..','keras-surgeon','src')
if not new_sys_dir in sys.path:
    sys.path.append(new_sys_dir)

from kerassurgeon import identify,utils
from kerassurgeon.operations import delete_channels
from kerassurgeon import Surgeon

import cv2
def get_images(image_dir,image_lists, preprocess_fct):
    images=[]
    y=[]
    class_ind=0
    for label in image_lists:
        img_files = sorted(image_lists[label]['training'])
        shuffle(img_files)
        for f in img_files[:5]:
            #print(os.path.join(image_dir,image_lists[label]['dir'],f))
            img=cv2.imread(os.path.join(image_dir,image_lists[label]['dir'],f))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            images.append(preprocess_fct(img.astype(np.float32)))
            y.append(class_ind)
        class_ind+=1
    images=np.array(images)
    y=np.array(y)
    print(images.shape,y.shape)
    return images,y

def get_model_apoz(model, image_dir,image_lists, preprocess_fct):
    # Get APoZ
    start = None
    end = None
    apoz = []
    index=0
    for layer in model.layers[start:end]:
        if is_conv(layer):
            print(layer.name)
            images,_=get_images(image_dir,image_lists, preprocess_fct)

            print(images.shape)
            apoz.extend([(layer.name, i, value) for (i, value)
                         in enumerate(get_apoz(model, layer, images))]) #val_images[:1000]
            index+=1
            #if index>=2:
            #    break

    layer_name, index, apoz_value = zip(*apoz)
    apoz_df = pd.DataFrame({'layer': layer_name, 'index': index,
                            'apoz': apoz_value})
    apoz_df = apoz_df.set_index('layer')
    return apoz_df

def get_channels_apoz_importance(model, layer, x_val, node_indices=None):
    if isinstance(layer, str):
        layer = model.get_layer(name=layer)

    # Check that layer is in the model
    if layer not in model.layers:
        raise ValueError('layer is not a valid Layer in model.')

    layer_node_indices = utils.find_nodes_in_model(model, layer)
    # If no nodes are specified, all of the layer's inbound nodes which are
    # in model are selected.
    if not node_indices:
        node_indices = layer_node_indices
    # Check for duplicate node indices
    elif len(node_indices) != len(set(node_indices)):
        raise ValueError('`node_indices` contains duplicate values.')
    # Check that all of the selected nodes are in the layer
    elif not set(node_indices).issubset(layer_node_indices):
        raise ValueError('One or more nodes specified by `layer` and '
                         '`node_indices` are not in `model`.')

    data_format = getattr(layer, 'data_format', 'channels_last')
    # Perform the forward pass and get the activations of the layer.
    mean_calculator = utils.MeanCalculator(sum_axis=0)
    print('layer:',layer,layer_node_indices,node_indices)
    
    for node_index in node_indices:
        act_layer, act_index = utils.find_activation_layer(layer, node_index)
        print('act layer',act_layer, act_index)
        # Get activations
        if hasattr(x_val, "__iter__"):
            temp_model = Model(model.inputs,
                               act_layer.get_output_at(act_index))
            print('before: act output',act_layer.get_output_at(act_index))
            a = temp_model.predict(x_val)
            #a=temp_model.predict_generator(x_val, x_val.n // x_val.batch_size)
            print('after:',layer,a.shape)
        else:
            get_activations = K.function(
                [single_element(model.inputs), K.learning_phase()],
                [act_layer.get_output_at(act_index)])
            a = get_activations([x_val, 0])[0]
            # Ensure that the channels axis is last
        if data_format == 'channels_first':
            a = np.swapaxes(a, 1, -1)
        # Flatten all except channels axis
        activations = np.reshape(a, [-1, a.shape[-1]])
        zeros = (activations == 0).astype(int)
        mean_calculator.add(zeros)

    return mean_calculator.calculate()

    
def get_channels_importance(model, layer, x_val, y , node_indices=None):
    if isinstance(layer, str):
        layer = model.get_layer(name=layer)

    # Check that layer is in the model
    if layer not in model.layers:
        raise ValueError('layer is not a valid Layer in model.')

    layer_node_indices = utils.find_nodes_in_model(model, layer)
    print('layer:',layer,layer_node_indices)
    # If no nodes are specified, all of the layer's inbound nodes which are
    # in model are selected.
    if not node_indices:
        node_indices = layer_node_indices
    # Check for duplicate node indices
    elif len(node_indices) != len(set(node_indices)):
        raise ValueError('`node_indices` contains duplicate values.')
    # Check that all of the selected nodes are in the layer
    elif not set(node_indices).issubset(layer_node_indices):
        raise ValueError('One or more nodes specified by `layer` and '
                         '`node_indices` are not in `model`.')

    data_format = getattr(layer, 'data_format', 'channels_last')
    # Perform the forward pass and get the activations of the layer.
    importances=[]
    print('layer:',layer,layer_node_indices,node_indices)
    if len(node_indices)>1:
        print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!')
    # Get activations
    if hasattr(x_val, "__iter__"):
        temp_model = Model(model.inputs,layer.output)
        print('before: act output',layer.output)
        a = temp_model.predict(x_val)
        #a=temp_model.predict_generator(x_val, x_val.n // x_val.batch_size)
    if data_format == 'channels_first':
        a = np.swapaxes(a, 1, -1)
    print('after:',layer,a.shape,data_format)
    # Flatten all except channels axis
    for channel in range(a.shape[-1]):
        activations = a[...,channel]
        activations = np.reshape(activations, [activations.shape[0],-1])
        #print('after:',layer,activations.shape)
        #pair_dist=activations
        pair_dist=pairwise_distances(activations)
        #pair_dist/=pair_dist.mean()
        if False:
            importance=(abs(pair_dist)).sum()
        elif False:
            indices=np.argsort(pair_dist,axis=1)
            same_class=(y[indices[:,1:]]==y.reshape(y.shape[0],1))
            other_indices=indices[:,1:]
            first_same_class=np.argmax(same_class,axis=1)
            first_other_class=np.argmin(same_class,axis=1)
            #importance=sum([pair_dist[i,other_indices[i,first_same_class[i]]]/pair_dist[i,other_indices[i,first_other_class[i]]] for i in range(y.shape[0])])/y.shape[0]
            
            #importance=sum([pair_dist[i,other_indices[i,first_same_class[i]]] for i in range(y.shape[0])])/y.shape[0]
            importance=sum([pair_dist[i,other_indices[i,first_other_class[i]]] for i in range(y.shape[0])])/y.shape[0]
        else:
            classes=np.unique(y)
            num_classes=classes.shape[0]
            delta=len(y)//num_classes
            #class_dists=np.array([[np.median(pair_dist[y==classes[i]][:,y==classes[j]]) for j in range(num_classes)] for i in range(num_classes)])
            #class_dists=np.array([[pair_dist[y==classes[i]][:,y==classes[j]][np.where(pair_dist[y==classes[i]][:,y==classes[j]]!=0)].mean() for j in range(num_classes)] for i in range(num_classes)])
            #class_dists=np.array([[np.mean(pair_dist[i:i+delta,j:j+delta]) for j in range(0,len(y),delta)] for i in range(0,len(y),delta)])
            pdr=pair_dist.reshape(num_classes,delta,num_classes,delta)
            
            class_dists=np.median(pdr,axis=(1,3))
            #class_dists=np.sum(pdr,axis=(1,3))/np.sum(pdr>0,axis=(1,3))
            
            #instance_dists=np.array([[pair_dist[y==y[i]][:,y==y[j]][np.where(pair_dist[y==y[i]][:,y==y[j]]!=0)].mean() for j in range(y.shape[0])] for i in range(y.shape[0])])
            #instance_dists=np.array([[np.median(pair_dist[y==y[i]][:,y==y[j]]) for j in range(y.shape[0])] for i in range(y.shape[0])])
            #instance_dists=np.array([[np.median(pair_dist[y==y[i]][:,y==y[j]]) for j in range(y.shape[0])] for i in range(y.shape[0])])
            
            #instance_dists=np.array([[class_dists[y[i]][y[j]] for j in range(y.shape[0])] for i in range(y.shape[0])])
            instance_dists=np.repeat(np.repeat(class_dists,delta,axis=0),delta,axis=1)
            importance=-(((pair_dist-instance_dists)**2)/instance_dists).sum() #+np.log(instance_dists)
            #if abs(importance)<0.01:
            #    print(channel,pair_dist,instance_dists)
        
        importances.append(importance)
        #print(indices,y[indices])
        #print(first_same_class,first_other_class)
        #print(pair_dist)
        #sys.exit(0)

    importances=np.array(importances)
    return importances

    
def get_channels_loss(model, layer, x_val, y , node_indices=None):
    if isinstance(layer, str):
        layer = model.get_layer(name=layer)

    # Check that layer is in the model
    if layer not in model.layers:
        raise ValueError('layer is not a valid Layer in model.')

    layer_node_indices = utils.find_nodes_in_model(model, layer)
    print('layer:',layer,layer_node_indices)
    # If no nodes are specified, all of the layer's inbound nodes which are
    # in model are selected.
    if not node_indices:
        node_indices = layer_node_indices
    # Check for duplicate node indices
    elif len(node_indices) != len(set(node_indices)):
        raise ValueError('`node_indices` contains duplicate values.')
    # Check that all of the selected nodes are in the layer
    elif not set(node_indices).issubset(layer_node_indices):
        raise ValueError('One or more nodes specified by `layer` and '
                         '`node_indices` are not in `model`.')

    data_format = getattr(layer, 'data_format', 'channels_last')
    # Perform the forward pass and get the activations of the layer.
    importances=[]
    print('layer:',layer,layer_node_indices,node_indices)
    if len(node_indices)>1:
        print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!')
    # Get activations
    if hasattr(x_val, "__iter__"):
        temp_model = Model(model.inputs,layer.output)
        func=K.function([layer.output,model.input], [model.output])
        print('before: act output',layer.output)
        a = temp_model.predict(x_val)
        print(a.shape,layer.output.shape)
        #a=temp_model.predict_generator(x_val, x_val.n // x_val.batch_size)
    print('after:',layer,a.shape,data_format)
    # Flatten all except channels axis
    for channel in range(a.shape[-1]):
        activations = a[...,channel]
        #print('after:',layer,activations.shape)
        if True:
            a_new=a.copy()
            
            if data_format == 'channels_first':
                mean_activation=abs(a_new[:,channel]).mean()
                a_new[:,channel]=0
            else:
                mean_activation=abs(a_new[...,channel]).mean()
                a_new[...,channel]=0

        else:
            a_new=np.zeros(a.shape)
            
            if data_format == 'channels_first':
                mean_activation=abs(a[:,channel]).mean()
                a_new[:,channel]=a[:,channel]
            else:
                mean_activation=abs(a[...,channel]).mean()
                a_new[...,channel]=a[...,channel]

        y_pred=[]
        acc=0
        loss=0
        
        delta=128
        for i in range(0,a_new.shape[0],delta):
            x=a_new[i:i+delta]
            
            b=func([x,x_val[i:i+delta]])[0]
            #print(x.shape,b.shape,b[y[i]])
            #y_pred.extend(b)
            
            acc+=(np.argmax(b,axis=1)==y[i:i+delta]).sum()
            ind=np.meshgrid(np.arange(b.shape[1]),np.arange(b.shape[0]))[0]
            loss-=np.log(b[ind==y[i:i+delta].reshape(x.shape[0],1)]).sum()
            #for j in range(b.shape[0]):
            #    loss-=math.log(b[j][y[i+j]])
            #loss-=np.log().sum()
            #if np.argmax(b)==y[i]:
            #    acc+=1
            #loss-=math.log(b[y[i]])
        y_pred=np.array(y_pred)
        acc/=a_new.shape[0]
        loss/=a_new.shape[0]
        print(channel, 'y_pred:',y_pred.shape,acc,loss,mean_activation)

        importances.append(loss)
        #print(indices,y[indices])
        #print(pair_dist)
        #sys.exit(0)
    #sys.exit(0)

    importances=np.array(importances)
    return importances

def get_channels_gradients(model, layer, x_val, y , node_indices=None):
    if isinstance(layer, str):
        layer = model.get_layer(name=layer)

    # Check that layer is in the model
    if layer not in model.layers:
        raise ValueError('layer is not a valid Layer in model.')

    layer_node_indices = utils.find_nodes_in_model(model, layer)
    print('layer:',layer,layer_node_indices)
    # If no nodes are specified, all of the layer's inbound nodes which are
    # in model are selected.
    if not node_indices:
        node_indices = layer_node_indices
    # Check for duplicate node indices
    elif len(node_indices) != len(set(node_indices)):
        raise ValueError('`node_indices` contains duplicate values.')
    # Check that all of the selected nodes are in the layer
    elif not set(node_indices).issubset(layer_node_indices):
        raise ValueError('One or more nodes specified by `layer` and '
                         '`node_indices` are not in `model`.')

    data_format = getattr(layer, 'data_format', 'channels_last')
    # Perform the forward pass and get the activations of the layer.
    importances=[]
    print('layer:',layer,layer_node_indices,node_indices)
    if len(node_indices)>1:
        print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!')
    # Get activations
    if hasattr(x_val, "__iter__"):
        grads = K.gradients(model.total_loss, layer.output)[0]
        input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
        ]
        if False:
            mul_a_grads_tensor=K.mean(layer.output,axis=0)*K.mean(grads,axis=0)
            if K.image_data_format() != 'channels_first':
                x = K.permute_dimensions(mul_a_grads_tensor, (2, 0, 1))
            
            x_shape=K.int_shape(x)
            #print(x_shape)
            x=K.reshape(x,(x_shape[0],x_shape[1]*x_shape[2]))
            x=K.sum(x,axis=1)
            x=K.abs(x)
        else:
            mul_a_grads_tensor=layer.output*grads
            if K.image_data_format() == 'channels_first':
                x = K.permute_dimensions(mul_a_grads_tensor, (1, 0, 2, 3))
            else:
                x = K.permute_dimensions(mul_a_grads_tensor, (3, 0, 1, 2))
            
            x_shape=K.int_shape(x)
            #print(x_shape)
            x=K.reshape(x,(x_shape[0],-1,x_shape[2]*x_shape[3]))
            x=K.sum(x,axis=2)
            x=K.abs(x)
            x=K.sum(x,axis=1)
        func=K.function(input_tensors, [x])
        print('before: act output',layer.output)
        delta=32
        importances=None
        for i in range(0,x_val.shape[0],delta):
            x=x_val[i:i+delta]
            q_part= func([x,np.ones(x.shape[0]),y[i:i+delta],0])[0]
            
            if importances is None:
                importances=q_part.copy()
            else:
                importances+=q_part
        print('after:',importances.shape,layer.output.shape,data_format)
    return importances
    
def get_channels_importance_with_gradient(model, layer, x_val, y, node_indices=None):
    if isinstance(layer, str):
        layer = model.get_layer(name=layer)

    # Check that layer is in the model
    if layer not in model.layers:
        raise ValueError('layer is not a valid Layer in model.')

    layer_node_indices = utils.find_nodes_in_model(model, layer)
    print('layer:',layer,layer_node_indices)
    # If no nodes are specified, all of the layer's inbound nodes which are
    # in model are selected.
    if not node_indices:
        node_indices = layer_node_indices
    # Check for duplicate node indices
    elif len(node_indices) != len(set(node_indices)):
        raise ValueError('`node_indices` contains duplicate values.')
    # Check that all of the selected nodes are in the layer
    elif not set(node_indices).issubset(layer_node_indices):
        raise ValueError('One or more nodes specified by `layer` and '
                         '`node_indices` are not in `model`.')

    data_format = getattr(layer, 'data_format', 'channels_last')
    # Perform the forward pass and get the activations of the layer.
    importances=[]
    print('layer:',layer,layer_node_indices,node_indices)
    if len(node_indices)>1:
        print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!')
    # Get activations
    if hasattr(x_val, "__iter__"):
        temp_model = Model(model.inputs,layer.output)
        print('before: act output',layer.output)
        a = temp_model.predict(x_val)
        
        grads = K.gradients(model.total_loss, layer.output)[0]
        input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
        ]
        if K.image_data_format() == 'channels_first':
            dimensions=(1, 0, 2, 3)
        else:
            dimensions=(3, 0, 1, 2)
        
        acts = K.permute_dimensions(layer.output, dimensions)
        grads = K.permute_dimensions(grads, dimensions)
        
        grads_shape=K.int_shape(grads)
        #print(x_shape)
        grads=K.reshape(grads,(grads_shape[0],-1,grads_shape[2]*grads_shape[3]))
        grads=K.sum(K.abs(grads),axis=2)
        
        acts=K.reshape(acts,(grads_shape[0],-1,grads_shape[2]*grads_shape[3]))
        func=K.function(input_tensors, [acts,grads])
        
        print('before: act output',layer.output)
        delta=32
        importances=None
        for i in range(0,x_val.shape[0],delta):
            x_part=x_val[i:i+delta]
            y_part=y[i:i+delta]
            a,g= func([x_part,np.ones(x_part.shape[0]),y_part,0])
            #print('after:',a.shape,g.shape,data_format)
            
            num_channels=a.shape[0]
            if importances is None:
                importances=np.zeros(num_channels)
            for channel in range(num_channels):
                activations = a[channel]
                activations = np.reshape(activations, [activations.shape[0],-1])
                #print('after:',layer,activations.shape)
                #pair_dist=activations
                pair_dist=pairwise_distances(activations)
                weighted_pair_dist=pair_dist*np.transpose(g[channel])
                if True:
                    importance=(abs(weighted_pair_dist)).sum()
                else:
                    indices=np.argsort(weighted_pair_dist,axis=1)
                    
                    same_class=(y_part[indices[:,1:]]==y_part.reshape(y_part.shape[0],1))
                    other_indices=indices[:,1:]
                    first_same_class=np.argmax(same_class,axis=1)
                    first_other_class=np.argmin(same_class,axis=1)
                    #importance=sum([pair_dist[i,other_indices[i,first_same_class[i]]]/pair_dist[i,other_indices[i,first_other_class[i]]] for i in range(y.shape[0])])/y.shape[0]
                    
                    #importance=sum([pair_dist[i,other_indices[i,first_same_class[i]]] for i in range(y_part.shape[0])])/y_part.shape[0]
                    importance=sum([weighted_pair_dist[i,other_indices[i,first_other_class[i]]] for i in range(y_part.shape[0])])/y_part.shape[0]
                
                importances[channel]+=importance
        
        print('after:',importances.shape,layer.output.shape,data_format)
        #sys.exit(0)

    return importances

def get_channels_l1_norm(model, layer, node_indices=None):
    if isinstance(layer, str):
        layer = model.get_layer(name=layer)

    # Check that layer is in the model
    if layer not in model.layers:
        raise ValueError('layer is not a valid Layer in model.')

    layer_node_indices = utils.find_nodes_in_model(model, layer)
    print('layer:',layer,layer_node_indices)
    # If no nodes are specified, all of the layer's inbound nodes which are
    # in model are selected.
    if not node_indices:
        node_indices = layer_node_indices
    # Check for duplicate node indices
    elif len(node_indices) != len(set(node_indices)):
        raise ValueError('`node_indices` contains duplicate values.')
    # Check that all of the selected nodes are in the layer
    elif not set(node_indices).issubset(layer_node_indices):
        raise ValueError('One or more nodes specified by `layer` and '
                         '`node_indices` are not in `model`.')

    data_format = getattr(layer, 'data_format', 'channels_last')
    w=layer.get_weights()[0]
    if data_format == 'channels_first':
        w = np.swapaxes(w, 1, -1)
    importances=abs(w).sum(axis=(0,1,2))
    print(w.shape,importances.shape)
    return importances

def prune_model_by_layer(model, percent_channels_delete,image_dir,image_lists, preprocess_fct):
    start = None
    end = None
    #model.summary()
    # Create the Surgeon and add a 'delete_channels' job for each layer
    # whose channels are to be deleted.
    surgeon = Surgeon(model, copy=True)
    for layer in model.layers[start:end]:
        if identify.is_conv(layer):
            print(layer.name)
            
            num_total_channels=layer.output_shape[-1]
            num_removed_channels=int(num_total_channels*percent_channels_delete/100)//16*16
            if num_removed_channels>0:
                if False:
                    images,y=get_images(image_dir,image_lists, preprocess_fct)
                    
                    #channels_importance=get_channels_apoz_importance(model, layer, images)
                    #channels_importance=get_channels_importance(model, layer, images,y)
                    #channels_importance=get_channels_loss(model, layer, images,y)
                    #channels_importance=get_channels_gradients(model, layer, images,y)
                    #channels_importance=get_channels_importance_with_gradient(model, layer, images,y)
                else:
                    channels_importance=get_channels_l1_norm(model, layer)
                
                total_channels_sorted=np.argsort(channels_importance)
                #print('channels_importance:',channels_importance.shape,total_channels_sorted[:5])
                channels=total_channels_sorted[:num_removed_channels]
                print('before add_job:',layer.name,channels,channels_importance[channels],len(channels),num_total_channels)
                surgeon.add_job('delete_channels', layer,channels=channels)
    # Delete channels
    return surgeon.operate()

def prune_model_random(model, percent_channels_delete):
    start = None
    end = None
    # Create the Surgeon and add a 'delete_channels' job for each layer
    # whose channels are to be deleted.
    surgeon = Surgeon(model, copy=True)
    for layer in model.layers[start:end]:
        if identify.is_conv(layer):
            print(layer.name)
            num_total_channels=layer.output_shape[-1]
            total_channels = list(range(num_total_channels))
            shuffle(total_channels)
            num_removed_channels=int(num_total_channels*percent_channels_delete/100)//16*16
            if num_removed_channels>0:
                channels=total_channels[:num_removed_channels]
                print('before add_job:',layer.name,channels,len(channels),num_total_channels)
                surgeon.add_job('delete_channels', layer,channels=channels)
            
    # Delete channels
    return surgeon.operate()

    
  


BATCH_SIZE=32 #32 #8 #16


import tensorflow as tf
from tensorflow.python.platform import gfile
#from keras.applications.inception_v1 import InceptionV1
from keras.applications import vgg19, inception_v3, resnet50,inception_resnet_v2, resnet_v2, mobilenetv2, mobilenet
from keras.layers import Flatten, Dense, Dropout,GlobalAveragePooling2D,AveragePooling2D, Activation, Conv2D, Lambda, Input, Reshape
from keras.models import Model,load_model,model_from_json
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import regularizers
from keras.preprocessing.image import (ImageDataGenerator, Iterator,
                                       array_to_img, img_to_array, load_img)
from keras import backend as K



def as_bytes(bytes_or_text, encoding='utf-8'):
    """Converts bytes or unicode to `bytes`, using utf-8 encoding for text.

    # Arguments
        bytes_or_text: A `bytes`, `str`, or `unicode` object.
        encoding: A string indicating the charset for encoding unicode.

    # Returns
        A `bytes` object.

    # Raises
        TypeError: If `bytes_or_text` is not a binary or unicode string.
    """
    if isinstance(bytes_or_text, six.text_type):
        return bytes_or_text.encode(encoding)
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text
    else:
        raise TypeError('Expected binary or unicode string, got %r' %
                        (bytes_or_text,))


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
        self.num_class = len(classes)
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
            batch_y = np.zeros((len(batch_x), self.num_class),
                               dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

def create_image_lists(image_dir, train_count):
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
        #train_cnt=int(math.ceil(train_percent*len(file_list)))
        #print(label_name,train_percent,len(file_list),train_cnt)
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


def get_generators(image_lists, image_dir,preprocess_fct):
    train_datagen = CustomImageDataGenerator(rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
      #rescale=1. / 255)
      preprocessing_function=preprocess_fct)

    test_datagen = CustomImageDataGenerator(preprocessing_function=preprocess_fct)

    train_generator = train_datagen.flow_from_image_lists(
        image_lists=image_lists,
        category='training',
        image_dir=image_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',seed=RANDOM_SEED)

    validation_generator = test_datagen.flow_from_image_lists(
        image_lists=image_lists,
        category='validation',
        image_dir=image_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',seed=RANDOM_SEED)

    return train_generator, validation_generator

def get_model(classes_num):
    if False:
        basemodel = vgg19
        basemodel_class = basemodel.VGG19
    elif False:
        basemodel = inception_v3
        basemodel_class = basemodel.InceptionV3
    elif False:
        basemodel=mobilenet
        basemodel_class = basemodel.MobileNet
        net_model = basemodel_class(weights='imagenet',include_top=False,input_shape=(IMG_SIZE, IMG_SIZE, 3),pooling='avg')
    elif True:
        basemodel=mobilenetv2
        basemodel_class = basemodel.MobileNetV2
        net_model = basemodel_class(alpha=1.0,weights='imagenet',include_top=False,input_shape=(IMG_SIZE, IMG_SIZE, 3),pooling='avg')
    elif True:
        basemodel=inception_resnet_v2
        basemodel_class = basemodel.InceptionResNetV2
    elif True:
        basemodel = resnet_v2
        basemodel_class = basemodel.ResNet152V2
    else:
        basemodel = resnet50
        basemodel_class = basemodel.ResNet50
    
    if net_model is None:
        net_model = basemodel_class(weights='imagenet',include_top=False,input_shape=(IMG_SIZE, IMG_SIZE, 3),pooling='avg')
    
    last_model_layer = net_model.output
    #last_model_layer = GlobalAveragePooling2D()(last_model_layer)
    preds=Dense(classes_num, activation='softmax')(last_model_layer)
    f_model = Model(net_model.input, preds)

    return f_model,net_model,basemodel.preprocess_input

def save_model(model,filename):
    model.save_weights('weights.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('weights.h5')
    loaded_model.save(filename)
    
def train_model(f_model, base_model, generators, train_new_layers=True):
    train_generator, val_generator = generators
    nb_train_samples=train_generator.samples
    nb_validation_samples=val_generator.samples

    pretitle='caltech101/mobilenet_v2_1.0-l1_25-pruned'
    #pretitle='dogs/mobilenet_v1-l1_25-pruned'
    #pretitle='dogs/mobilenet_v1_l1reg-my25-pruned'
    mc = ModelCheckpoint(pretitle+'-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', verbose=1, save_best_only=True)
    
    if train_new_layers:
        base_model.trainable=False
        for l in base_model.layers:
            l.trainable=False
        f_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
        #f_model.summary()

        FIRST_EPOCHS=2
        #mc = ModelCheckpoint('dogs_mobilenet_multi_heads-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', verbose=1, save_best_only=True)
        hist1=f_model.fit_generator(train_generator, steps_per_epoch=nb_train_samples//BATCH_SIZE, epochs=FIRST_EPOCHS, verbose=1, initial_epoch=0, callbacks=[mc], validation_data=val_generator, validation_steps=nb_validation_samples // BATCH_SIZE)
        base_model.trainable=True
        for l in base_model.layers:
            l.trainable=True
        SECOND_EPOCHS=FIRST_EPOCHS+18
        initial_epoch=len(hist1.history['loss'])
    else:
        initial_epoch=20 #0 #8
        SECOND_EPOCHS=initial_epoch+20 #10 #2


    f_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])
    f_model.summary()

    es=EarlyStopping(monitor='val_acc',patience=2)
    hist1=f_model.fit_generator(train_generator, steps_per_epoch=nb_train_samples//BATCH_SIZE, epochs=SECOND_EPOCHS, verbose=1, 
                    initial_epoch=initial_epoch, callbacks=[mc,es], validation_data=val_generator, validation_steps=nb_validation_samples // BATCH_SIZE)
    #f_model.save(pretitle+'.h5')
    save_model(f_model,pretitle+'_short.h5')

    return f_model
    
    
import tempfile

def add_l1l2_regularizer(model, l1=0.0, l2=0.0, reg_attributes=None):
    # Add L1L2 regularization to the whole model.
    # NOTE: This will save and reload the model. Do not call this function inplace but with
    # model = add_l1l2_regularizer(model, ...)

    #for l in model.layers:
    #    print(l,l.losses)
        
    if not reg_attributes:
        reg_attributes = ['kernel_regularizer', 'bias_regularizer']
    if isinstance(reg_attributes, str):
        reg_attributes = [reg_attributes]

    regularizer = regularizers.l1(l1) #_l2(l1=l1, l2=l2)

    for layer in model.layers:
        for attr in reg_attributes:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # Reload the model
    model = model_from_json(model_json)
    model.load_weights(tmp_weights_path, by_name=True)
    #print('new')
    #for l in model.layers:
    #    print(l,l.losses)

    #sys.exit(0)
    return model
    
if __name__ == '__main__':
        
    #model_file='dogs/mobilenet_v1-20-0.67.h5'
    #model_file='dogs/mobilenet_v1_l1reg-40-0.72.h5'
    #model_file='caltech101/caltech101_mobilenet_v2_1.0-20-0.92.h5'
    model_file=''
    if False:
        file_name='caltech101_mobilenet_v2_1.4'
        model=load_model(file_name+'.h5')
        save_model(model,file_name+'_short.h5')
        sys.exit(0)

    image_dir='D:/datasets/caltech/101_ObjectCategories'
    train_count=30    
    #image_dir='D:/datasets/StanfordDogs'
    #train_count=70    
 
    
    image_lists,classes_num = create_image_lists(image_dir, train_count)
    print("Number of classes found: {}".format(classes_num))
    
    f_model,base_model,preprocess_fct = get_model(classes_num)
    generators = get_generators(image_lists, image_dir,preprocess_fct)

    train_new_layers=True

    if False:
        #model_files=['caltech101_mobilenet_v1_short','caltech101_mobilenet_v1-rnd25-pruned_short','caltech101_mobilenet_v2_1.0_short','caltech101_mobilenet_v2_1.0-rnd25-pruned_short','caltech101_mobilenet_v2_1.4_short','caltech101_mobilenet_v2_1.0-rnd25-pruned_short']
        model_files=['mobilenet_v1_l1reg_short','mobilenet_v1_l1reg-my25-pruned_short','mobilenet_v2_1.0_l1reg_base_short','mobilenet_v2_1.0_l1reg_my25-pruned_short','mobilenet_v2_1.4_short','mobilenet_v2_1.4_l1reg-my25-pruned_short']
        #basedir='caltech101_'
        base_dir='dogs/'
        #K.set_session(tf.Session(config=tf.ConfigProto(device_count = {'GPU' : 0})))
        for model_file in model_files:
            f_model=load_model(base_dir+model_file+'.h5') #'caltech101_mobilenet.h5') #-pruned
            f_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])
            train_new_layers=False
            if True:
                x=np.random.uniform(-1,1,(1,IMG_SIZE,IMG_SIZE,3))
                preds=f_model.predict(x)[0]
                TESTS=100
                start_time=time.time()
                for _ in range(TESTS):
                    preds=f_model.predict(x)[0]
                elapsed=time.time()-start_time
                print(model_file,' elapsed (ms):',1000*elapsed/TESTS,' size (Mb):',os.path.getsize(base_dir+model_file+'.h5')/(1024*1024.0))
            elif True:
                val_generator=generators[1]
                score = f_model.evaluate_generator(val_generator, val_generator.samples//BATCH_SIZE)
                print(model_file, ' evaluation scores:',score)
        sys.exit(0)
    elif os.path.exists(model_file):
        train_new_layers=False
        f_model.load_weights(model_file)
        #save_model(f_model,model_file+'_short.h5')
        #sys.exit(0)
        if True:
            f_model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])
            #f_model=prune_model_random(f_model, 25)
            f_model=prune_model_by_layer(f_model, 25,image_dir,image_lists, preprocess_fct)
        else:
            f_model=add_l1l2_regularizer(f_model, l1=0.0001, l2=0.0)


    f_model = train_model(f_model, base_model,generators=generators, train_new_layers=train_new_layers)
    