# Tensorflow VGG Face

Face features of [VGG FACE [0]](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/). The code is based on [Caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow)


#### Requirements
- Tensorflow in your python environment.
- Caffe in CAFFE_HOME (For vgg_face preparations and tests)
- Download [LFW [1]](http://vis-www.cs.umass.edu/lfw/) dataset (optional, for more testing, TODO: it's not giving the reported result yet)

### Setup
- Download `git clone --recursive https://github.com/po0ya/tensorflow_vgg_face`
- Link two directories named `data` and `debug` in the root of the project
- Run `$ export CAFFE_HOME=path/to/caffe`
- Run `$ ./scripts/prepare_vgg_face.sh` to download VGG_FACE and upgrade the prototxts

### Testing
- Create your face images list and run `$ ./scripts/get_feats.py path/to/img_list path/to/save/features`
- To double check the features run `$ ./scripts/test.sh`
- In `config.py` if `cfg.FLIP = True` the images are flipped and the features are averaged as explained in [0]
- In `config.py` if `cfg.CROP = True` images are cropped in 5 ways instead of just center according to [0] and features are averaged
- Face bounding boxes can be saved into `{im_path: [x1,y1,x2,y2]}` format for `im_path in img_list` and fed to the test script
as in `./scripts/eval_lfw.sh`

##### LFW
- Link downloaded LFW dataset to `data/lfw` directory. Move all the images to `data/lfw/imgs`. Create the face bounding box dictionary 
and `cPickle.dump` it to `data/lfw/imgs/img_list_bboxes.pckl`.
- Run `$ ./scripts/eval_lfw.sh` 
- TODO: Verification results on LFW are low.

## References
 [0]  O. M. Parkhi, A. Vedaldi, A. Zisserman, Deep Face Recognition, British Machine Vision Conference, 2015 
 
 [1] Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. University of Massachusetts, Amherst, Technical Report 07-49, October, 2007.
