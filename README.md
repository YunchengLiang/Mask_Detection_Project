# Mask_Detection_Project
## Support both image and video real time face mask detection
Two-Stage Model Architecture:
  1. Using pretrained SSD to detect faces
  2. Once faces are detected, feed into mobilenet-v2 finetuned for classification of "face with mask" vs "face without mask" 

