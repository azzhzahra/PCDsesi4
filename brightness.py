import imageio as img 
import numpy as np
import matplotlib.pyplot as plt

def bright(image,factor):
    bright_image = image.astype(np.float32)
    bright_image += bright_image + factor   
    
    bright_image = np.clip(bright_image,0,255)
    
    return bright_image.astype(np.uint8)   

image = img.imread("C:\\Users\\komputer 35\\Downloads\\download (2).jpg")
br_image = bright(image,50) 

plt.figure(figsize=(15,8))
plt.subplot(1,2,1) 
plt.imshow(image) 
plt.subplot(1,2,2) 
plt.imshow(br_image) 
plt.show()