import sys
import os.path
import os
import numpy as np
import time,datetime

VALID_IMAGE_FORMATS = frozenset(['jpg', 'jpeg'])
IMG_SIZE=299
#IMG_SIZE=224
BATCH_SIZE=8
finetune=True


from keras.applications import vgg19, inception_v3, resnet50,inception_resnet_v2, resnet_v2, mobilenet,mobilenet_v2
from keras.models import Model,load_model
from keras.preprocessing import image


def get_model():    
    net_model=None
    if False:
        basemodel = vgg19
        basemodel_class = basemodel.VGG19
    elif False:
        basemodel=mobilenet
        basemodel_class = basemodel.MobileNet
    elif True:
        basemodel = inception_v3
        basemodel_class = basemodel.InceptionV3
        IMG_SIZE=299
    elif True:
        basemodel=inception_resnet_v2
        basemodel_class = basemodel.InceptionResNetV2
        IMG_SIZE=299
    elif False:
        basemodel=mobilenet_v2
        basemodel_class = basemodel.MobileNetV2
    elif True:
        basemodel = resnet_v2
        basemodel_class = basemodel.ResNet152V2
    else:
        basemodel = resnet50
        basemodel_class = basemodel.ResNet50
    
    if basemodel==mobilenet_v2:
        net_model = basemodel_class(alpha=1.4,weights='imagenet',include_top=False,input_shape=(IMG_SIZE, IMG_SIZE, 3),pooling='avg')
    else:
        net_model = basemodel_class(weights='imagenet',include_top=False,input_shape=(IMG_SIZE, IMG_SIZE, 3),pooling='avg')

    return basemodel,net_model

def extract_features_from_batch(model,preprocess_input,images):
    x = np.array(images)
    x = preprocess_input(x)
    all_features = model.predict(x)
    return all_features

def write_batch(all_features,filenames,d,outF):
    for fn,features in zip(filenames,all_features):
        print(os.path.join(d,fn), file=outF)
        print(d, file=outF)
        for f in features:
           outF.write('{:f} '.format(f)) 
        outF.write("\n")


import os,sys
import json
import numpy as np
import tensorflow as tf

TPU_MODELS_PATH='D:/src_code/tpu/models'
sys.path.append(os.path.join(TPU_MODELS_PATH,'official/efficientnet'))
sys.path.append(os.path.join(TPU_MODELS_PATH,'common'))

import efficientnet_builder
import preprocessing
import utils

class EfficientNet():
    def __init__(self,model_name='efficientnet-b0',ckpt_dir=None):
        self.model_name = model_name
        self.ckpt_dir=ckpt_dir if ckpt_dir is not None else model_name
        _, _, self.image_size, _ = efficientnet_builder.efficientnet_params(model_name)
        
        with tf.Graph().as_default() as graph:
            self.sess=tf.Session()
            self.filename=tf.placeholder(tf.string)
            image_string = tf.read_file(self.filename)
            preprocess_fn = self.get_preprocess_fn()
            image_decoded = preprocess_fn(image_string, False, image_size=self.image_size)
            image = tf.cast(image_decoded, tf.float32)
            images=tf.expand_dims(image, 0)
            self.probs = self.build_model(images)
            self.restore_model()

    def restore_model(self, enable_ema=True):
        """Restore variables from checkpoint dir."""
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(os.path.join(TPU_MODELS_PATH,'official/efficientnet/checkpoints/',self.ckpt_dir))
        if enable_ema:
            ema = tf.train.ExponentialMovingAverage(decay=0.0)
            ema_vars = utils.get_ema_vars()
            var_dict = ema.variables_to_restore(ema_vars)
            ema_assign_op = ema.apply(ema_vars)
        else:
            var_dict = get_ema_vars()
            ema_assign_op = None

        tf.train.get_or_create_global_step()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_dict, max_to_keep=1)
        saver.restore(self.sess, checkpoint)
          
    def build_model(self, images):
        images -= tf.constant(
          efficientnet_builder.MEAN_RGB, shape=[1, 1, 3], dtype=images.dtype)
        images /= tf.constant(
          efficientnet_builder.STDDEV_RGB, shape=[1, 1, 3], dtype=images.dtype)
        features, _ = efficientnet_builder.build_model(
            images, self.model_name, False, pooled_features_only=True)
        return features

    def get_preprocess_fn(self):
        return preprocessing.preprocess_image
    
    def extract_features(self,image_file):
        return self.sess.run(self.probs, feed_dict={self.filename:image_file})

USE_EFFICIENT_NET=True
    
if __name__ == '__main__':
    image_dir='D:/datasets/caltech/256_ObjectCategories'
    #image_dir='D:/datasets/StanfordDogs/dogs_all'
    VALID_IMAGE_FORMATS = frozenset(['.jpg', '.jpeg'])

    if USE_EFFICIENT_NET:
        ckpt_dir=None
        model_name='efficientnet-b0'
        #ckpt_dir='efficientnet-b7-randaug'
        net=EfficientNet(model_name=model_name, ckpt_dir=ckpt_dir)
    else:
        basemodel,model=get_model()
        preprocess_input=basemodel.preprocess_input
        model_name=basemodel.__name__.replace('keras.applications.','')
    
    out_filename=os.path.join('precomputed_features',os.path.basename(image_dir)+'_'+model_name+'.txt')
    print(out_filename)
    with open(out_filename, "w") as outF:
        for d in next(os.walk(image_dir))[1]:
            print(d)
            dir_path = os.path.join(image_dir, d)
            filenames,images=[],[]
            for f in next(os.walk(dir_path))[2]:
                _, file_extension = os.path.splitext(f)
                if file_extension.lower() in VALID_IMAGE_FORMATS:
                    filenames.append(f)
                    if USE_EFFICIENT_NET:
                        try:
                            all_features=net.extract_features(os.path.join(dir_path,f))
                            write_batch(all_features,filenames,d,outF)                        
                        except Exception as e:
                            print('invalid file ',os.path.join(dir_path,f), e)
                        filenames=[]
                    else:
                        img = image.load_img(os.path.join(dir_path,f), target_size=(IMG_SIZE,IMG_SIZE))
                        x = image.img_to_array(img)
                        images.append(x)
                        if len(filenames)>32:
                            all_features=extract_features_from_batch(model,preprocess_input,images)
                            write_batch(all_features,filenames,d,outF)
                            filenames,images=[],[]
            
            if not USE_EFFICIENT_NET and len(filenames)>0:
                all_features=extract_features_from_batch(model,preprocess_input,images)
                write_batch(all_features,filenames,d,outF)
    print(all_features.shape)
