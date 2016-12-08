'''Utility functions and classes for handling image datasets.'''
import cPickle
import os.path as osp
import numpy as np
import tensorflow as tf
from config import cfg

FLAGS = tf.app.flags.FLAGS


def process_image_adv(img, scale, isotropic, crop, mean,flip=False,crop_ind=0,face_bbox=None):
    '''Crops, scales, and normalizes the given image.
    scale : The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
    crop  : After scaling, depending on crop_ind a crop of the image is given.
    crope_ind: 0 center, 1 SW, 2 SE, 3 NE, 4 NW crop
    flip: Whether to flip the image
    mean  : Subtracted from the image
    '''

    if face_bbox is not None:
        img = tf.slice(img, begin=tf.pack([face_bbox[0], face_bbox[1], 0]), size=tf.pack([face_bbox[2]-face_bbox[0], face_bbox[3]-face_bbox[1], -1]))

    # Rescale
    if flip:
        img = tf.reverse(img,[False,True,False])

    if isotropic:
        img_shape = tf.to_float(tf.shape(img)[:2])
        min_length = tf.minimum(img_shape[0], img_shape[1])
        new_shape = tf.to_int32((scale / min_length) * img_shape)
    else:
        new_shape = tf.pack([scale, scale])
    img = tf.image.resize_images(img, new_shape)
    offset = [0,0]
    if crop_ind == 1:
        offset[0] = new_shape[0]-crop
        offset = new_shape-crop
    elif crop_ind == 2:
        offset = new_shape-crop
    elif crop_ind == 3:
        offset[1] = new_shape[1]-crop
    elif crop_ind == 4:
        offset = [0,0]
    else:
        offset = (new_shape - crop) / 2

    img = tf.slice(img, begin=tf.pack([offset[0], offset[1], 0]), size=tf.pack([crop, crop, -1]))
    # Mean subtraction
    return tf.to_float(img) - mean

def process_image(img, scale, isotropic, crop, mean):
    '''Crops, scales, and normalizes the given image.
    scale : The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
    crop  : After scaling, a central crop of this size is taken.
    mean  : Subtracted from the image
    '''
    # Rescale
    if isotropic:
        img_shape = tf.to_float(tf.shape(img)[:2])
        min_length = tf.minimum(img_shape[0], img_shape[1])
        new_shape = tf.to_int32((scale / min_length) * img_shape)
    else:
        new_shape = tf.pack([scale, scale])
    img = tf.image.resize_images(img, new_shape)
    # Center crop
    # Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
    # See: https://github.com/tensorflow/tensorflow/issues/521
    offset = (new_shape - crop) / 2
    img = tf.slice(img, begin=tf.pack([offset[0], offset[1], 0]), size=tf.pack([crop, crop, -1]))
    # Mean subtraction
    return tf.to_float(img) - mean


class ImageProducer(object):
    '''
    Loads and processes batches of images in parallel.
    '''

    def __init__(self, image_paths, data_spec, num_concurrent=4, batch_size=None, labels=None):
        # The data specifications describe how to process the image
        self.data_spec = data_spec
        # A list of full image paths
        self.image_paths = image_paths
        # An optional list of labels corresponding to each image path
        self.labels = labels
        # A boolean flag per image indicating whether its a JPEG or PNG
        self.extension_mask = self.create_extension_mask(self.image_paths)
        # Create the loading and processing operations
        self.setup(batch_size=batch_size, num_concurrent=num_concurrent)

    def start(self, session, coordinator, num_concurrent=4):
        '''Start the processing worker threads.'''
        # Queue all paths
        session.run(self.enqueue_paths_op)
        # Close the path queue
        session.run(self.close_path_queue_op)
        # Start the queue runner and return the created threads
        return self.queue_runner.create_threads(session, coord=coordinator, start=True)

    def get(self, session):
        '''
        Get a single batch of images along with their indices. If a set of labels were provided,
        the corresponding labels are returned instead of the indices.
        '''
        (indices, images) = session.run(self.dequeue_op)
        if self.labels is not None:
            labels = [self.labels[idx] for idx in indices]
            return (labels, images)
        return (indices, images)

    def batches(self, session):
        '''Yield a batch until no more images are left.'''
        for _ in xrange(self.num_batches):
            yield self.get(session=session)

    def load_image(self, image_path, is_jpeg):
        # Read the file
        file_data = tf.read_file(image_path)
        # Decode the image data
        img = tf.cond(
            is_jpeg,
            lambda: tf.image.decode_jpeg(file_data, channels=self.data_spec.channels),
            lambda: tf.image.decode_png(file_data, channels=self.data_spec.channels))
        if self.data_spec.expects_bgr:
            # Convert from RGB channel ordering to BGR
            # This matches, for instance, how OpenCV orders the channels.
            img = tf.reverse(img, [False, False, True])
        return img

    def process(self,crop_flip):
        # Dequeue a single image path
        idx, is_jpeg, image_path = self.path_bbox_queue.dequeue()
        # Load the image
        # Process the image
        img_list = []
        idx_list = []
        for (c,f) in crop_flip:
            img = self.load_image(image_path, is_jpeg)

            processed_img = process_image_adv(img=img,
                                          scale=self.data_spec.scale_size,
                                          isotropic=self.data_spec.isotropic,
                                          crop=self.data_spec.crop_size,
                                          mean=self.data_spec.mean,
                                              flip=f,
                                              crop_ind=c)
            img_list.append(processed_img)
            idx_list.append(idx)
        # Return the processed image, along with its index

        processed_idx_list = tf.pack(idx_list)
        processed_img_list = tf.pack(img_list)
        return (processed_idx_list, processed_img_list)

    @staticmethod
    def create_extension_mask(paths):

        def is_jpeg(path):
            extension = osp.splitext(path)[-1].lower()
            if extension in ('.jpg', '.jpeg'):
                return True
            if extension != '.png':
                raise ValueError('Unsupported image format: {}'.format(extension))
            return False

        return [is_jpeg(p) for p in paths]

    def __len__(self):
        return len(self.image_paths)

    def setup(self, batch_size, num_concurrent):
        pass


class VGGFaceProducer(ImageProducer):

    def __init__(self, image_paths, data_spec ,num_concurrent=4,bbox_fp=None):
        round_rect = lambda x: [int(p) for p in x]
        try:
            v = self.face_bboxes
        except AttributeError:
            self.face_bboxes = None
            if bbox_fp is not None:
                face_bboxes=cPickle.load(open(bbox_fp,'r'))
                self.face_bboxes = [round_rect(face_bboxes[p][0]) for p in image_paths]

        # Initialize base
        super(VGGFaceProducer, self).__init__(image_paths=image_paths,
                                               data_spec=data_spec,num_concurrent=num_concurrent)

    def setup(self, batch_size, num_concurrent):
        # Validate the batch size
        num_images = len(self.image_paths)
        batch_size = min(num_images, batch_size or self.data_spec.batch_size)
        if num_images % batch_size != 0:
            raise ValueError(
                'The total number of images ({}) must be divisible by the batch size ({}).'.format(
                    num_images, batch_size))
        self.num_batches = num_images / batch_size

        # Create a queue that will contain image paths (and their indices and extension indicator)
        if self.face_bboxes is None:
            self.path_bbox_queue = tf.FIFOQueue(capacity=num_images,
                                            dtypes=[tf.int32, tf.bool, tf.string],
                                            name='path_queue')
            indices = tf.range(num_images)
            self.enqueue_paths_op = self.path_bbox_queue.enqueue_many([indices, self.extension_mask,
                                                                   self.image_paths])
        else:
            self.path_bbox_queue = tf.FIFOQueue(capacity=num_images,
                                                dtypes=[tf.int32, tf.bool, tf.string, tf.int32],
                                                name='path_queue')
            indices = tf.range(num_images)
            self.enqueue_paths_op = self.path_bbox_queue.enqueue_many([indices, self.extension_mask,
                                                                                                  self.image_paths,self.face_bboxes])
        # Close the path queue (no more additions)
        self.close_path_queue_op = self.path_bbox_queue.close()

        # Create an operation that dequeues a single path and returns a processed image
        crop_flip = [[0,False]]
        if cfg.CROP:
            for i in range(1,5):
                crop_flip.append([i,False])

        if cfg.FLIP:
            for i in range(len(crop_flip)):
                crop_flip.append((crop_flip[i][0],True))

        (processed_idx_list,processed_img_list) = self.process(crop_flip)
        # Create a queue that will contain the processed images (and their indices)
        image_shape = (self.data_spec.crop_size, self.data_spec.crop_size, self.data_spec.channels)
        processed_queue = tf.FIFOQueue(capacity=int(np.ceil(len(crop_flip)*num_images / float(num_concurrent))),
                                       dtypes=[tf.int32, tf.float32],
                                       shapes=[(), image_shape],
                                       name='processed_queue')

        # Enqueue the processed image and path
        enqueue_processed_op = processed_queue.enqueue_many([processed_idx_list,processed_img_list])

        # Create a dequeue op that fetches a batch of processed images off the queue
        [self.ind_deq,self.img_deq] = processed_queue.dequeue_many(batch_size)
        self.dequeue_op = [self.ind_deq,self.img_deq]

        # Create a queue runner to perform the processing operations in parallel
        num_concurrent = min(num_concurrent, num_images)
        self.queue_runner = tf.train.QueueRunner(processed_queue,
                                                 [enqueue_processed_op] * num_concurrent)

        self.num_imgs = len(crop_flip)*num_images
        self.num_feats_per_image = len(crop_flip)


    def process(self,crop_flip):
        # Dequeue a single image path
        if self.face_bboxes is None:
            idx, is_jpeg, image_path = self.path_bbox_queue.dequeue()
            face_bbox = None
        else:
            idx, is_jpeg, image_path, face_bbox = self.path_bbox_queue.dequeue()
        # Load the image
        # Process the image
        img_list = []
        idx_list = []
        for (c,f) in crop_flip:
            img = self.load_image(image_path, is_jpeg)

            processed_img = process_image_adv(img=img,
                                          scale=self.data_spec.scale_size,
                                          isotropic=self.data_spec.isotropic,
                                          crop=self.data_spec.crop_size,
                                          mean=self.data_spec.mean,
                                              flip=f,
                                              crop_ind=c,
                                              face_bbox=face_bbox)
            img_list.append(processed_img)
            idx_list.append(idx)
        # Return the processed image, along with its index

        processed_idx_list = tf.pack(idx_list)
        processed_img_list = tf.pack(img_list)
        return (processed_idx_list, processed_img_list)

class LFWProducer(VGGFaceProducer):
    def __init__(self, val_path, data_path, data_spec,bbox_fp=None,num_concurrent=4):
        round_rect = lambda x: [int(p) for p in x]
        image_paths = [osp.join(data_path, p) for p in val_path]
        self.face_bboxes=None
        if bbox_fp is not None:
            face_bboxes=cPickle.load(open(bbox_fp,'r'))
            self.face_bboxes = [round_rect(face_bboxes[p][0]) for p in val_path]
        super(LFWProducer, self).__init__(image_paths=image_paths,
                                               data_spec=data_spec,num_concurrent=num_concurrent)

