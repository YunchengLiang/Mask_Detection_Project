import os 
import sys
import random
import numpy as np
import argparse
from PIL import Image, ImageFile
from pathlib import Path

mask_base_path=r"C:\Users\87032\OneDrive\Documents\GitHub\Mask_Detection_Project\Data_Generator\images\masks"
mask_paths=[ os.path.join(mask_base_path,i) for i in os.listdir(mask_base_path)]

def create_mask(image_path):
    pic_path=image_path
    random_num=np.random.randint(len(mask_paths))
    print(random_num)
    mask_path= mask_paths[random_num] #randomly choose a mask to wear
    show=False
    model="hog"
    FaceMasker(pic_path,mask_path,show,model).mask()

class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')
    def __init__(self, face_path, mask_path, show=False, model='hog'):
        self.face_path=face_path
        self.mask_path=mask_path
        self.show=show
        self.model=model
    def mask(self):
        import face_recognition
        face_image_np = face_recognition.load_image_file(self.face_path) # convert image to numpy
        face_locations= face_recognition.face_locations(face_image_np, model=self.model) #may have mutiple location of faces
        face_landmarks= face_recognition.face_landmarks(face_image_np, face_locations) #extract landmarks for faces detected 
        self._face_img= Image.fromarray(face_image_np)
        self._mask_img= Image.open(self.mask_path)

        found_face= False
        #for every face detected, detect in landmarks whether both the 'nose bridge' and the 'chin' is there or not, 
        # if not, skip, we do not need to add mask for those skipped 
        for face_landmark in face_landmarks:
            skip=False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    print(facial_feature)
                    skip=True
                    break
            if skip:
                continue
            # mask the face 
            found_face=True
            self._mask_face(face_landmark)
        if found_face:
            if self.show:
                self._face_img.show()
            self._save()
        else:
            print("Does not find face")

    def _mask_face(self, face_landmark:dict):
        nose_bridge= face_landmark['nose_bridge']
        nose_point= nose_bridge[1] 
        nose_v= np.array(nose_point)   

        chin= face_landmark['chin']
        chin_len= len(chin)
        chin_bottom_point= chin[8]
        chin_bottom_v= np.array(chin_bottom_point)
        chin_left_point= chin[2]
        chin_right_point= chin[14]    

        #split mask and resize
        width= self._mask_img.width
        height= self._mask_img.height
        width_ratio = 1.2 #real size mask is wider than the face, so it could cover the face
        new_height = int(np.linalg.norm(nose_v-chin_bottom_v))

        #left side of mask
        mask_left_img = self._mask_img.crop((0,0, width//2,  height))
        mask_left_width= self.get_distance_from_point_to_line(chin_left_point,nose_point,chin_bottom_point)
        mask_left_width= int(mask_left_width*width_ratio)
        mask_left_img= mask_left_img.resize((mask_left_width,new_height))

        #right side of mask
        mask_right_img = self._mask_img.crop((width//2, 0, width, height))
        mask_right_width= self.get_distance_from_point_to_line(chin_right_point,nose_point,chin_bottom_point)
        mask_right_width= int(mask_right_width*width_ratio)
        mask_right_img= mask_right_img.resize((mask_right_width,new_height))

        #merge mask
        size= (mask_left_img.width + mask_right_img.width, new_height)
        mask_img= Image.new('RGBA',size)
        mask_img.paste(mask_left_img,(0,0),mask_left_img)
        mask_img.paste(mask_right_img,(mask_left_img.width,0),mask_right_img) #image.paste(logo, postion, logo) logo appears again for transparency settings

        #rotate mask
        angle= np.arctan2(chin_bottom_point[1]-nose_point[1], chin_bottom_point[0]-nose_point[0]) #!!!!
        rotated_mask_img= mask_img.rotate(angle=angle, expand=True)

        #calculate mask location
        center_x= (nose_point[0]+chin_bottom_point[0])//2
        center_y= (nose_point[1]+chin_bottom_point[1])//2
        offset=mask_img.width//2 - mask_left_img.width
        radian= angle*np.pi/180
        box_x= center_x+ int(offset*np.cos(radian)) - rotated_mask_img.width //2
        box_y= center_y+ int(offset*np.sin(radian)) - rotated_mask_img.height //2

        #add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)

    def _save(self):
        new_face_name="add_mask\with_mask_" +os.path.basename(self.face_path)
        p = os.path.join(Path(self.face_path).parent,new_face_name)
        print(p)
        self._face_img.save(p)
        print(f'saved to {p}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        point=np.array(point)
        line_point1=np.array(line_point1)
        line_point2=np.array(line_point2)
        return abs(np.cross(line_point2-line_point1,point-line_point1)/np.linalg.norm(line_point2-line_point1))

if __name__ == '__main__':
    create_mask(image_path)   

          