import cv2
from generate_mask import create_mask


img=cv2.imread('/dev/shm/Places2/val/val_256/Places365_val_00000001.jpg')
mask=create_mask()
mask=cv2.bitwise_not(mask)
print(mask.shape)
print(img.shape)
cv2.imwrite('mask.jpg',mask)
img_masked=cv2.bitwise_and(img,img,mask=mask)
cv2.imwrite('test.jpg',img_masked)
mask=mask/255
print(mask)