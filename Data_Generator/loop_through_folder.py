import cv2 
import os 
from mask import create_mask

folder_path = r"C:\Users\87032\OneDrive\Documents\GitHub\Mask_Detection_Project\Data_Generator\Downloads"

images= [os.path.join(folder_path,i) for i in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,i))]
for j in range(len(images)):
    print("Image path: ", images[j])
    create_mask(images[j])