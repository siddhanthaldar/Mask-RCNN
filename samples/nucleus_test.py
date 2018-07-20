import os
import sys
import random
import math
import json
import datetime
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_nucleus.h5")  # Path to nucleus
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class NucleusDataset(utils.Dataset):

	def load_nucleus(self, dataset_dir, subset):
		"""Load a subset of the nuclei dataset.

		dataset_dir: Root directory of the dataset
		subset: Subset to load. Either the name of the sub-directory,
		        such as stage1_train, stage1_test, ...etc. or, one of:
		        * train: stage1_train excluding validation images
		        * val: validation images from VAL_IMAGE_IDS
		"""
		# Add classes. We have one class.
		# Naming the dataset nucleus, and the class nucleus
		self.add_class("nucleus", 1, "nucleus")

		# Which subset?
		# "val": use hard-coded list above
		# "train": use data from stage1_train minus the hard-coded list above
		# else: use the data from the specified sub-directory
		assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
		subset_dir = "stage1_train" if subset in ["train", "val"] else subset
		dataset_dir = os.path.join(dataset_dir, subset_dir)
		if subset == "val":
			image_ids = VAL_IMAGE_IDS
		else:
			# Get image ids from directory names
			image_ids = next(os.walk(dataset_dir))[1]
			if subset == "train":
				image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

		# Add images
		for image_id in image_ids:
			self.add_image(
				"nucleus",
				image_id=image_id,
				path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))



class InferenceConfig(coco.CocoConfig):
	"""Configuration for training on the nucleus segmentation dataset."""
	# Give the configuration a recognizable name
	NAME = "nucleus"

	# Adjust depending on your GPU memory
	IMAGES_PER_GPU = 4 #6

	# Number of classes (including background)
	NUM_CLASSES = 1 + 1  # Background + nucleus

	# Don't exclude based on confidence. Since we have two classes
	# then 0.5 is the minimum anyway as it picks between nucleus and BG
	DETECTION_MIN_CONFIDENCE = 0

	# Backbone network architecture
	# Supported values are: resnet50, resnet101
	BACKBONE = "resnet50"

	# Input image resizing
	# Random crops of size 512x512
	IMAGE_RESIZE_MODE = "crop"
	IMAGE_MIN_DIM = 256 #512
	IMAGE_MAX_DIM = 256 #512
	IMAGE_MIN_SCALE = 2.0

	# Length of square anchor side in pixels
	RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

	# ROIs kept after non-maximum supression (training and inference)
	POST_NMS_ROIS_TRAINING = 500 #1000
	POST_NMS_ROIS_INFERENCE = 1000 #2000

	# Non-max suppression threshold to filter RPN proposals.
	# You can increase this during training to generate more propsals.
	RPN_NMS_THRESHOLD = 0.9

	# How many anchors per image to use for RPN training
	RPN_TRAIN_ANCHORS_PER_IMAGE = 64

	# Image mean (RGB)
	MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

	# If enabled, resizes instance masks to a smaller size to reduce
	# memory load. Recommended when using high-resolution images.
	USE_MINI_MASK = True
	MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

	# Number of ROIs per image to feed to classifier/mask heads
	# The Mask RCNN paper uses 512 but often the RPN doesn't generate
	# enough positive proposals to fill this and keep a positive:negative
	# ratio of 1:3. You can increase the number of proposals by adjusting
	# the RPN NMS threshold.
	TRAIN_ROIS_PER_IMAGE = 64 #128

	# Maximum number of ground truth instances to use in one image
	MAX_GT_INSTANCES = 100 #200

	# Max number of final detections per image
	DETECTION_MAX_INSTANCES = 200 #400

	# Set batch size to 1 to run one image at a time
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	# Don't resize imager for inferencing
	IMAGE_RESIZE_MODE = "pad64"
	# Non-max suppression threshold to filter RPN proposals.
	# You can increase this during training to generate more propsals.
	RPN_NMS_THRESHOLD = 0.7

    

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'nucleus']
'''
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread("/home/sid/virtual_env/deep_learning/codes/clones/Mask_RCNN/images/nucleus3.jpg")
# Run detection
results = model.detect([image], verbose=1)

# Visualize results
print("\nReached\n")
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
'''

############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
	"""Encodes a mask in Run Length Encoding (RLE).
	Returns a string of space-separated values.
	"""
	assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
	# Flatten it column wise
	m = mask.T.flatten()
	# Compute gradient. Equals 1 or -1 at transition points
	g = np.diff(np.concatenate([[0], m, [0]]), n=1)
	# 1-based indicies of transition points (where gradient != 0)
	rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
	# Convert second index in each pair to lenth
	rle[:, 1] = rle[:, 1] - rle[:, 0]
	return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
	"""Decodes an RLE encoded list of space separated
	numbers and returns a binary mask."""
	rle = list(map(int, rle.split()))
	rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
	rle[:, 1] += rle[:, 0]
	rle -= 1
	mask = np.zeros([shape[0] * shape[1]], np.bool)
	for s, e in rle:
	    assert 0 <= s < mask.shape[0]
	    assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
	    mask[s:e] = 1
	# Reshape and transpose
	mask = mask.reshape([shape[1], shape[0]]).T
	return mask


def mask_to_rle(image_id, mask, scores):
	"Encodes instance masks to submission format."
	assert mask.ndim == 3, "Mask must be [H, W, count]"
	# If mask is empty, return line with image ID only
	if mask.shape[-1] == 0:
		return "{},".format(image_id)
	# Remove mask overlaps
	# Multiply each instance mask by its score order
	# then take the maximum across the last dimension
	order = np.argsort(scores)[::-1] + 1  # 1-based descending
	mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
	# Loop over instance masks
	lines = []
	for o in order:
		m = np.where(mask == o, 1, 0)
		# Skip if empty
		if m.sum() == 0.0:
			continue
		rle = rle_encode(m)
		lines.append("{},{}".format(image_id, rle))	
	return "\n".join(lines)


def detect(model, dataset_dir, subset):
	"""Run detection on images in the given directory."""
	print("Running on {}".format(dataset_dir))

	# Create directory
	if not os.path.exists(RESULTS_DIR):
	    os.makedirs(RESULTS_DIR)
	submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
	submit_dir = os.path.join(RESULTS_DIR, submit_dir)
	os.makedirs(submit_dir)

	# Read dataset
	dataset = NucleusDataset()
	dataset.load_nucleus(dataset_dir, subset)
	dataset.prepare()
	# Load over images
	submission = []
	count = 0
	for image_id in dataset.image_ids:
		count += 1
		print(count)
		# Load image and run detection
		image = dataset.load_image(image_id)
		# Detect objects
		r = model.detect([image], verbose=0)[0]
		# Encode image to RLE. Returns a string of multiple lines
		source_id = dataset.image_info[image_id]["id"]
		rle = mask_to_rle(source_id, r["masks"], r["scores"])
		submission.append(rle)
		# Save image with masks
		visualize.display_instances(
		    image, r['rois'], r['masks'], r['class_ids'],
		    dataset.class_names, r['scores'],
		    show_bbox=False, show_mask=False,
		    title="Predictions")
		# plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

	# Save to csv file
	submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
	file_path = os.path.join(submit_dir, "stage2_test_submit.csv")
	with open(file_path, "w") as f:
		f.write(submission)
	print("Saved to ", submit_dir)

dataset_dir = os.path.join(ROOT_DIR,"samples/nucleus/data")
detect(model, dataset_dir, "stage2_test")