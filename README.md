# Mask_Detection_Project
## Support both image and video real time face mask detection end-to-end:
  1. For local image face mask detection, use detect_mask_from_image.py
  2. For local video streaming face mask detection, use detect_mask_from_video.py
  3. The website for above two functions was built using Flask framework,    
     for local standalone test, run 'python app.py',    
     for deployment on Heroku, change the filename under .github/workflows to main.yaml and replace the "email, heroku_api_key, heroku_app_name" to one's own
## Two-Stage Model Architecture:
  1. Using pretrained SSD to detect faces
  2. Once faces are detected, feed into mobilenet-v2 finetuned for classification of "face with mask" vs "face without mask" 
## Dataset & Data Augmentation:
  1. Dataset from https://www.kaggle.com/datasets/omkargurav/face-mask-dataset, contains folder of two categories (with_mask, without_mask), around 7000 images
  2. Besides data augmentation using ImageDataGenerator, in the folder called Data_Generator, augmentation method of using opencv to add masks to unmasked faces in the picture was performed and added to the "with mask" dataset
## Artifacts:
  1. mask_detector.model is stage 2 mask classfication model finetuned based on mobilenet-v2
  2. plot.png is the model training loss and accuracy history
  3. demoimage_XXX are demo images inferenced by the model with faces with/without masks highlighted with bounding box and confidence
  4. demovideo.mp4 are demo for video streaming using local computer's camera
  5. examples folder contains demo images for image face mask detection
  6. face_detector folder contains pretrained SSD model for face detection and its weights
  7. In Data_Generator/images/masks, there are the masks used for augmenting the with_mask datasets
  8. In Data_Generator/Downloads/add_mask, there are the augumented with_mask faces using masks, they are then added to the dataset/with_mask folder
## Requirements:
  pip install -r requirements.txt

