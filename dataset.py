import cv2
import numpy as np
import glob
import os

x = glob.glob("osm2world_500_HR_WS/hudsonriver500/*.jpg")
for file in x:
    filename = "Dataset/train/HI/" + os.path.basename(file) 
    image = cv2.imread(file)
    height, width = image.shape[:2]
    print (image.shape)
    # Let's get the starting pixel coordiantes (top left of cropped top)
    start_row, start_col = int(0), int(0)
    # Let's get the ending pixel coordinates (bottom right of cropped top)
    end_row, end_col = int(height), int(width * .5)
    cropped_top = image[start_row:end_row , start_col:end_col]
    print (start_row, end_row) 
    print (start_col, end_col)
    cv2.imwrite(filename, cropped_top)
    filename = "Dataset/train/GT/" + os.path.basename(file) 
    # Let's get the starting pixel coordiantes (top left of cropped bottom)
    start_row, start_col = int(0), int(width * .5)
    # Let's get the ending pixel coordinates (bottom right of cropped bottom)
    end_row, end_col = int(height), int(width)
    cropped_bot = image[start_row:end_row , start_col:end_col]
    print (start_row, end_row) 
    print (start_col, end_col)
    cv2.imwrite(filename, cropped_bot)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
x = glob.glob("osm2world_500_HR_WS/wallstreet500/*.jpg")
for file in x:
    filename = "Dataset/valid/HI/" + os.path.basename(file) 
    image = cv2.imread(file)
    height, width = image.shape[:2]
    print (image.shape)
    # Let's get the starting pixel coordiantes (top left of cropped top)
    start_row, start_col = int(0), int(0)
    # Let's get the ending pixel coordinates (bottom right of cropped top)
    end_row, end_col = int(height), int(width * .5)
    cropped_top = image[start_row:end_row , start_col:end_col]
    print (start_row, end_row) 
    print (start_col, end_col)
    cv2.imwrite(filename, cropped_top)
    filename = "Dataset/valid/GT/" + os.path.basename(file) 
    # Let's get the starting pixel coordiantes (top left of cropped bottom)
    start_row, start_col = int(0), int(width * .5)
    # Let's get the ending pixel coordinates (bottom right of cropped bottom)
    end_row, end_col = int(height), int(width)
    cropped_bot = image[start_row:end_row , start_col:end_col]
    print (start_row, end_row) 
    print (start_col, end_col)
    cv2.imwrite(filename, cropped_bot)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    #cv2.imshow("Original Image", image)
    #cv2.waitKey(0) 