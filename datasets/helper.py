#MODIFIED CAFFE_TENSORFLOW
import numpy as np
from models.vgg.vgg_face_embedding import VGG_FACE_16

class DataSpec(object):
    '''Input data specifications for an ImageNet model.'''

    def __init__(self,
                 batch_size,
                 scale_size,
                 crop_size,
                 isotropic,
                 channels=3,
                 mean=None,
                 bgr=True):
        # The recommended batch size for this model
        self.batch_size = batch_size
        # The image should be scaled to this size first during preprocessing
        self.scale_size = scale_size
        # Whether the model expects the rescaling to be isotropic
        self.isotropic = isotropic
        # A square crop of this dimension is expected by this model
        self.crop_size = crop_size
        # The number of channels in the input image expected by this model
        self.channels = channels
        # The mean to be subtracted from each image. By default, the per-channel ImageNet mean.
        # The values below are ordered BGR, as many Caffe models are trained in this order.
        # Some of the earlier models (like AlexNet) used a spatial three-channeled mean.
        # However, using just the per-channel mean values instead doesn't affect things too much.
        self.mean = mean if mean is not None else np.array([104., 117., 124.])
        # Whether this model expects images to be in BGR order
        self.expects_bgr = True


def alexnet_spec(batch_size=500):
    '''Parameters used by AlexNet and its variants.'''
    return DataSpec(batch_size=batch_size, scale_size=256, crop_size=227, isotropic=False)


def std_spec(batch_size, isotropic=True):
    '''Parameters commonly used by "post-AlexNet" architectures.'''
    return DataSpec(batch_size=batch_size, scale_size=256, crop_size=224, isotropic=isotropic)

# Collection of sample auto-generated models
MODELS = (VGG_FACE_16,)

# The corresponding data specifications for the sample models
# These specifications are based on how the models were trained.
# The recommended batch size is based on a Titan X (12GB).
MODEL_DATA_SPECS = {
    VGG_FACE_16:DataSpec(batch_size=10, scale_size=256, crop_size=224, isotropic=True,mean=np.array([93.5940,104.7624,129.1863])),
}


def get_models():
    '''Returns a tuple of sample models.'''
    return MODELS


def get_data_spec(model_instance=None, model_class=None):
    '''Returns the data specifications for the given network.'''
    model_class = model_class
    return MODEL_DATA_SPECS[model_class]
