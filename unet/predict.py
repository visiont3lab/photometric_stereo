from config import *
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from model import UNET

imresize = transforms.Compose([
    transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
])

def make_predictions(model, imagePath):
    model.eval()
    with torch.no_grad():

        image = Image.open(imagePath).convert('RGB')
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        image = imresize(image)
        image = transforms.ToTensor()(image).to(DEVICE)

        image = image.unsqueeze(0)
        predMask = model(image)
        predMask = predMask.squeeze(0).cpu().numpy()

        # Show the images
        cv2.imshow("predMask", predMask[0])
        cv2.imshow("image", open_cv_image)
        cv2.waitKey(0)

if __name__ == '__main__':
    # load the image paths in our testing file and randomly select 10
    # image paths
    print("[INFO] loading up test image paths...")
    imagePaths = open(TEST_PATHS).read().strip().split("\n")
    imagePaths = np.random.choice(imagePaths, size=10)
    unet = UNET(3, 3)
    unet.load_state_dict(torch.load(MODEL_PATH))
    unet = unet.to(DEVICE)
    unet.eval()

    for path in imagePaths:
        make_predictions(unet, path)