import os 
import argparse
import numpy as np
import cv2
def get_list_image(name_dataset,path):
    if name_dataset =="Places2":
        path=path+"/train"
        list_image=[]
        print(os.listdir(path))
        for x in os.listdir(path):
            x=os.path.join(path,x)
            print(x, flush=True)
            for y in os.listdir(x):
                y=os.path.join(x,y)
                for z in os.listdir(y):
                    z=os.path.join(y,z)
                    if os.path.isfile(z):
                        list_image.append(z)
                    else:
                        for image in os.listdir(z):
                            image=os.path.join(z,image)
                            list_image.append(image)
    else:
        pass

    return list_image

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Your description here')
    parser.add_argument('--name', type=str, help='Name of the dataset', required=True)
    parser.add_argument('--path', type=str, help='Path to the dataset file', required=True)
    args = parser.parse_args()
    list_image=get_list_image(args.name,args.path)
    total_pixels = 0
    sum_pixels = np.zeros(3, dtype=np.float64)
    sum_squared_pixels = np.zeros(3, dtype=np.float64)
    for i, x in enumerate(list_image):
        if i % 10000 == 0:
            print(f"Processing image {i}", flush=True)
        image = cv2.imread(x)
        if image is None:
            print(f"Failed to read image: {x}")
            continue

        # Convert image to float64 and correct color space if needed
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image.reshape(-1, 3).astype(np.float64)/255
        # print(pixels, type(pixels))
        sum_pixels += np.sum(pixels, axis=0)/pixels.shape[0]
        # print(sum_pixels,type(sum_pixels))
        # print(pixels.shape[0])
        # break
        sum_squared_pixels += np.sum(np.square(pixels), axis=0)/pixels.shape[0]
        # total_pixels += pixels.shape[0]

    
    mean = sum_pixels / len(list_image)
    variance = (sum_squared_pixels / len(list_image)) - np.square(mean)
    # print((sum_squared_pixels / total_pixels))
    print(np.square(mean))
    print(variance)
    std = np.sqrt(variance)

    np.savez(os.path.join(args.path,"mean.npz"), mean)
    np.savez(os.path.join(args.path,"std.npz"),std)




