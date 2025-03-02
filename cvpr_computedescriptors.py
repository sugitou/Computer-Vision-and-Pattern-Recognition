import os
import cv2.xfeatures2d
import numpy as np
import cv2
from scipy.ndimage import convolve
import scipy.io as sio

DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
OUT_FOLDER = 'descriptors'
OUT_SUBFOLDER = 'globalRGBhisto_27'  # 'globalRGBhisto_27', 'globalRGBhisto_64', 'globalRGBhisto_125'
                                     # 'edgehisto_10', 'edgehisto_20', 'edgehisto_60'

descriptor = 'GlobalColourHistogram'  # 'GlobalColourHistogram', 'EdgeOrientationHistogram'
level_quntization = 4  # colour 3(27), 4(64), 5(125)
                        # edge 10, 20, 60


def extractDescriptor(img, descriptor, Q=level_quntization):
    if descriptor == 'GlobalColourHistogram':
        # Split the image into RGB channels and reshape each to a flat array
        R, G, B = img[:, :, 0].ravel(), img[:, :, 1].ravel(), img[:, :, 2].ravel()

        # Quantize the RGB values and create histogram bins
        R_index = np.floor(R * Q).astype(int)
        G_index = np.floor(G * Q).astype(int)
        B_index = np.floor(B * Q).astype(int)

        # Create a unique bin index for each quantized RGB triplet
        N = R_index * (Q**2) + G_index * Q + B_index
        bins = Q ** 3
        histogram, _ = np.histogram(N, bins=np.arange(bins + 1))
        
        F = histogram / np.sum(histogram)
        
        return F
                
    elif descriptor == 'EdgeOrientationHistogram':
        # Convert to grayscale
        # Convert to float32 if not already
        if img.dtype != np.float32:
            img = img.astype(np.float32)  
            
        gray_float32 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray_float32.astype(np.float64)
        
        # sobel kernels
        sobel_x_kernel = np.array([[1, 0, -1],
                                   [2, 0, -2],
                                   [1, 0, -1]])

        sobel_y_kernel = np.array([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]])

        sobel_x = convolve(gray, sobel_x_kernel)
        sobel_y = convolve(gray, sobel_y_kernel)
        
        # Compute gradient magnitude and direction
        mag = np.sqrt(np.multiply(sobel_x, sobel_x) + np.multiply(sobel_y, sobel_y))
        angle = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)
        angle = np.mod(angle, 180)
        # Compute the histogram
        local_histogram, _ = np.histogram(angle, bins=Q, range=(0, 180), weights=mag)
        
        F = local_histogram / np.sum(local_histogram)
    
    else:
        raise ValueError('Invalid descriptor')
    
    return F

# Ensure the output directory exists
os.makedirs(os.path.join(OUT_FOLDER, OUT_SUBFOLDER), exist_ok=True)

# Iterate through all BMP files in the dataset folder
filenames = os.listdir(os.path.join(DATASET_FOLDER, 'Images'))
for filename in filenames:
    if filename.endswith(".bmp"):
        print(f"Processing file {filename}")
        img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
        img = cv2.imread(img_path).astype(np.float64) / 255.0  # Normalize the image
        fout = os.path.join(OUT_FOLDER, OUT_SUBFOLDER, filename.replace('.bmp', '.mat'))
        
        # Call extractDescriptor to get the descriptor
        F = extractDescriptor(img, descriptor)
        
        # Save the descriptor to a .mat file
        sio.savemat(fout, {'F': F})