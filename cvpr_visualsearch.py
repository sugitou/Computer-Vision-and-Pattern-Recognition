import os
import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
from cvpr_distance import cvpr_distance
from cvpr_pca import pca_dimen_reduc

DESCRIPTOR_FOLDER = 'descriptors'
IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
OUT_IMAGE_FOLDER = 'result_images'
OUT_PR_FOLDER = 'result_PR'

num_bins = '64'  # '27', '64', '125'
                 # '10', '20', '60'
DESCRIPTOR_SUBFOLDER = 'globalRGBhisto' + '_' + num_bins  # 'globalRGBhisto', 'edgehisto'
metric='Mahalanobis'  # 'Euclidean', 'Mahalanobis', 'Cosine', 'Manhattan'
QUERY=2               # select Query image 0, 1, 2 
PCA=1                 # select PCA or not 0, 1
DIMENSION=42          # select the number of PCA dimensions

TEST_IMAGES = ['MSRC_ObjCategImageDatabase_v2/Images/1_28_s.bmp',
               'MSRC_ObjCategImageDatabase_v2/Images/10_8_s.bmp',
               'MSRC_ObjCategImageDatabase_v2/Images/20_10_s.bmp']


def create_filename(folder_name):
    # Ensure the output directory exists
    os.makedirs(folder_name, exist_ok=True)
    # Save the plot
    output_filename = f"{folder_name}/{metric}_{num_bins}-{querynames[QUERY].split('/')[-1].split('.')[0]}.png"
    
    return output_filename

def eval_precision(file_num):
    query_label = ALLFILES[queryimg].split('/')[-1].split('_')[0]
    candidate_label = ALLFILES[file_num].split('/')[-1].split('_')[0]
    if query_label == candidate_label:
        return True
    else:
        return False

# Load all descriptors
ALLFEAT = []
ALLFILES = []
for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
    if filename.endswith('.mat'):
        img_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
        img_actual_path = os.path.join(IMAGE_FOLDER,'Images',filename).replace(".mat",".bmp")
        img_data = sio.loadmat(img_path)
        ALLFILES.append(img_actual_path)
        ALLFEAT.append(img_data['F'][0])  # Assuming F is a 1D array
        
ALLFEAT = np.array(ALLFEAT)

if PCA == 1:
    ALLFEAT_pca = pca_dimen_reduc(ALLFEAT, DIMENSION)
    ALLFEAT = ALLFEAT_pca
else:
    pass

# Identify the query image
queryimgs = []
querynames = []

for i, file in enumerate(ALLFILES):
    for search_value in TEST_IMAGES:
        if search_value in file:
            queryimgs.append(i)
            querynames.append(file)

# number of img and adjust this value to check all query image
NIMG = ALLFEAT.shape[0]
queryimg = queryimgs[QUERY]

# Compute the distance between the query and all other descriptors
dst = []
query = ALLFEAT[queryimg]
# Compute the covariance matrix and its inverse
cov_mat_all = np.cov(ALLFEAT, rowvar=False)
inv_cov_mat_all = np.linalg.inv(cov_mat_all)

for i in range(NIMG):
    candidate = ALLFEAT[i]
    distance = cvpr_distance(query, candidate, metric=metric, inv_cov_mat=inv_cov_mat_all)
    dst.append((distance, i))

# Sort the distances
dst.sort(key=lambda x: x[0])

# Show the top 10 results
SHOW = 11
images = []

for i in range(SHOW):
    img = cv2.imread(ALLFILES[dst[i][1]])
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Make image quarter size
    images.append(img)

# create a figure and a set of subplots
fig = plt.figure()

# Set the plot arrangement
X = 3
Y = 4

for i, img in enumerate(images):
    imgplot = i + 1
    ax = fig.add_subplot(X, Y, imgplot)
    if i == 0:
        ax.set_title(f"Query img",fontsize=10)
    else:
        ax.set_title(f"Rank {i}",fontsize=10)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Save the plot
plt.savefig(create_filename(OUT_IMAGE_FOLDER))

plt.show()


# Prepare the query number for the precision-recall curve
cat = []
for A in ALLFILES:
    if A.split('/')[-1].split('_')[0] == ALLFILES[queryimg].split('/')[-1].split('_')[0]:
       cat.append(1) 

# Calc precision and recall 
precision, recall = [], []
r_true = 0
for i, t in enumerate(dst[1:NIMG]):
    i += 1
    if eval_precision(t[1]) == True:
        r_true += 1
        precision.append(r_true/i)
        recall.append(r_true/cat.count(1))
    else:
        precision.append(r_true/i)
        recall.append(r_true/cat.count(1))

# plot the precision-recall curve
plt.plot(recall, precision, label='PR curve', marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)

# Save the plot
plt.savefig(create_filename(OUT_PR_FOLDER))

plt.show()