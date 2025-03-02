import os
import numpy as np
import scipy.io as sio
import cv2
from cvpr_distance import cvpr_distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

DESCRIPTOR_FOLDER = 'descriptors'
num_bins = '60'  # '10', '20', '60'
                 # '27', '64', '125'
DESCRIPTOR_SUBFOLDER = 'edgehisto' + '_' + num_bins  # 'globalRGBhisto', 'edgehisto', 'BagofVisualWords'
IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'

'''
    Don't use Mahalanobis distance without PCA
'''
metric='Mahalanobis'  # 'Euclidean', 'Mahalanobis', 'Cosine', 'Manhattan'
QUERY=2               # select query image
PCA=1                 # select PCA or not
DIMENSION=40          # select the number of PCA dimensions

KMEANS=0              # select KMEANS or not
N_CLUSTERS=50         # select the number of clusters

TEST_IMAGES = ['MSRC_ObjCategImageDatabase_v2/Images/1_28_s.bmp',
               'MSRC_ObjCategImageDatabase_v2/Images/10_8_s.bmp',
               'MSRC_ObjCategImageDatabase_v2/Images/20_10_s.bmp']

def create_codebook(descriptors, num_clusters):
    # Create the KMeans object
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(descriptors)
    
    # Get the codebook
    codebook = kmeans.cluster_centers_
    
    return codebook

def pca_dimen_reduc(features, k=DIMENSION):
    # Perform PCA on all descriptors
    mean_F = np.mean(features, axis=0)
    centered_F = features - mean_F
    # Compute the covariance matrix
    cov_mat = np.cov(centered_F, rowvar=False)
    # Decompose the covariance matrix into eigenvectors and eigenvalues
    val, vct = np.linalg.eig(cov_mat)
    # Sort the eigenvalues and eigenvectors
    sorted_indices = np.argsort(val)[::-1]
    # sorted_val = val[sorted_indices]
    sorted_vct = vct[:, sorted_indices]
    # Select the top k eigenvectors
    projection_matrix = sorted_vct[:, :k]
    # Project the data onto the new space
    pca_result = np.dot(centered_F, projection_matrix)
    
    return pca_result

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
        if KMEANS == 1:
            ALLFEAT.append(img_data['F'])
        else:
            ALLFEAT.append(img_data['F'][0])  # Assuming F is a 1D array
        

if KMEANS == 1:
    all_descriptors = np.vstack(ALLFEAT)
    codebook = create_codebook(all_descriptors, N_CLUSTERS)
else:
    ALLFEAT = np.array(ALLFEAT)

if PCA == 1:
    ALLFEAT_pca = pca_dimen_reduc(ALLFEAT)
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
base_height = 200

for i in range(SHOW):
    img = cv2.imread(ALLFILES[dst[i][1]])
    # Aspect ratio resize
    # aspect_ratio = img.shape[1] / img.shape[0]
    # new_width = int(base_height * aspect_ratio)
    # img = cv2.resize(img, (new_width, base_height))
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Make image quarter size
    images.append(img)

# create a figure and a set of subplots
fig = plt.figure()

#flg全体をX*Yに分割し、plot位置に画像を配置する。
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
output_filename = f"result_images/{metric}_{num_bins}-{querynames[QUERY].split('/')[-1].split('.')[0]}.png"
plt.savefig(output_filename)

plt.show()
    
# # Conatenate all images vertically
# combined_image = np.hstack(images)

# # Save the combined image
# cv2.imwrite(output_filename, combined_image)

# # Show the combined image
# cv2.imshow("Combined Results", combined_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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
output_filename = f"result_PR/{metric}_{num_bins}-{querynames[QUERY].split('/')[-1].split('.')[0]}.png"
plt.savefig(output_filename)

plt.show()

# # Compute confusion matrix
# y_true = []
# for t in dst[:NIMG]:
#     if eval_precision(t[1]):
#         y_true.append(1)
#     else:
#         y_true.append(0)
# threadshold = 0.2
# y_pred = []
# for t in dst[:NIMG]:
#     if threadshold > t[0]:
#         y_pred.append(1)
#     else:
#         y_pred.append(0)
# y_true = np.array(y_true)
# y_pred = np.array(y_pred)
# cm = confusion_matrix(y_true, y_pred)

# # Plot confusion matrix as heatmap
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()