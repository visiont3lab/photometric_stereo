from dataset import SegmentationDataset
from model import UNET
from config import *
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import torch
import time
from tqdm import tqdm

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(
        s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

# define transformations
def_transforms = transforms.Compose([
        transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),        
    ]) 

class Trainer():

    def init_dataset(self):
        # load the image and mask filepaths in a sorted manner
        self.imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
        self.maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))
        split = train_test_split(self.imagePaths, self.maskPaths,
                                test_size=TEST_SPLIT, random_state=42)
        # unpack the data split
        (self.trainImages, self.testImages) = split[:2]
        (self.trainMasks, self.testMasks) = split[2:]
        # write the testing image paths to disk so that we can use then
        # when evaluating/testing our model
        print("[INFO] saving testing image paths...")
        f = open(TEST_PATHS, "w")
        f.write("\n".join(self.testImages))
        f.close()

        # create the train and test datasets
        trainDS = SegmentationDataset(imagePaths=self.trainImages, maskPaths=self.trainMasks,
                                    transforms=def_transforms)
        testDS = SegmentationDataset(imagePaths=self.testImages, maskPaths=self.testMasks,
                                    transforms=def_transforms)

        print(f"[INFO] found {len(trainDS)} examples in the training set...")
        print(f"[INFO] found {len(testDS)} examples in the test set...")

        # create the training and test data loaders
        self.trainLoader = DataLoader(trainDS, shuffle=True,
                                batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
                                num_workers=NUM_WORKERS)
        self.testLoader = DataLoader(testDS, shuffle=False,
                                batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
                                num_workers=NUM_WORKERS)
        # calculate steps per epoch for training and test set
        self.trainSteps = len(trainDS) // BATCH_SIZE
        self.testSteps = len(testDS) // BATCH_SIZE
        # initialize a dictionary to store training history
        self.H = {"train_loss": [], "test_loss": []}
        return True
    
    def init_model(self):
        # initialize our UNet model
        self.unet = UNET(3, 3).to(DEVICE)
        self.unet.eval()
        self.opt = Adam(self.unet.parameters(), lr=INIT_LR)
        self.lossFunc = torch.nn.MSELoss()
        print(self.unet)
   
    def __init__(self):
        self.init_dataset()
        self.init_model()
    
    def save_model(self):
        # serialize the model to disk
        torch.save(self.unet.state_dict(), MODEL_PATH)

    def launch_training(self):
        # loop over epochs
        print("[INFO] training the network...")
        startTime = time.time()
        loss = 0
        for e in tqdm(range(NUM_EPOCHS)):
            # set the model in training mode
            self.unet.train()
            # initialize the total training and validation loss
            totalTrainLoss = 0
            totalTestLoss = 0
            # loop over the training set
            for index,  (imagergb, labelmask) in enumerate(self.trainLoader):
                self.opt.zero_grad()
                # send the input to the device
                # x y 
                x = imagergb.to(DEVICE)
                y = labelmask.to(DEVICE)
                # perform a forward pass and calculate the training loss
                pred = self.unet(x)
                loss = self.lossFunc(pred, y)
                
                loss.backward()
                self.opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss += loss
            # switch off autograd
            with torch.no_grad():
                # set the model in evaluation mode
                self.unet.eval()
                # loop over the validation set
                for (imagergb, labelmask) in self.testLoader:
                    # send the input to the device
                    (x, y) = (imagergb.to(DEVICE), labelmask.to(DEVICE))
                    # make the predictions and calculate the validation loss
                    pred = self.unet(x)
                    totalTestLoss += self.lossFunc(pred, y)

            # calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / self.trainSteps
            avgTestLoss = totalTestLoss / self.testSteps
            # update our training history
            self.H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            self.H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
            print("Train loss: {:.6f}, Test loss: {:.4f}".format(
                avgTrainLoss, avgTestLoss))
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime))

if __name__ == '__main__':    
    force_cudnn_initialization()
    tensor = Trainer()
    tensor.launch_training()
    tensor.save_model()