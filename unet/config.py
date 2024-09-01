# import the necessary packages
import torch
import os

# base path of the dataset
DATASET_PATH = "../blender/dataset"
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "normals")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "depths")

# define the test split
TEST_SPLIT = 0.15

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# initialize learning rate, number of epochs to train for, and the
INIT_LR = 1e-5 # 0.0005
NUM_EPOCHS = 20
BATCH_SIZE = 10
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.999

# define the input image dimensions
INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_WIDTH = 256

# define threshold to filter weak predictions
THRESHOLD = 0.7

# define the path to the base output directory
BASE_OUTPUT = "output"
os.makedirs(BASE_OUTPUT, exist_ok=True)
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, f"unet_{INPUT_IMAGE_HEIGHT}.pth")
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])