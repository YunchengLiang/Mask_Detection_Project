# Mask_Detection_Project
## Support both image and video real time face mask detection:
  1. For image face mask detection, use detect_mask_from_image.py
  2. For video streaming face mask detection, use detect_mask_from_video.py
## Two-Stage Model Architecture:
  1. Using pretrained SSD to detect faces
  2. Once faces are detected, feed into mobilenet-v2 finetuned for classification of "face with mask" vs "face without mask" 
## Dataset & Data Augmentation:
  1. Dataset from https://www.kaggle.com/datasets/omkargurav/face-mask-dataset, contains folder of two categories (with_mask, without_mask), around 7000 images
  2. Besides data augmentation using ImageDataGenerator, in the folder called Data_Generator, augmentation method of using opencv to add masks to unmasked faces in the picture was performed and added to the "with mask" dataset

