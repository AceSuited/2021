import cv2
import numpy as np
import glob
import os
import sys

# outputs config
STUDENT_CODE = '2016112083'
FILE_NAME= 'output.txt'
if not os.path.exists(STUDENT_CODE):
    os.mkdir(STUDENT_CODE)
f = open(os.path.join(STUDENT_CODE, FILE_NAME), 'w')


###############################################
############# STEP 1 ##########################
###############################################

## get input percentage
percentage = float(sys.argv[1])

## dataset build
train_ds = np.array([cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob("faces_training/*.pgm"))])

## vectorize
train_ds = train_ds.reshape(-1, 192 * 168)

## centerize
centered = train_ds - np.mean(train_ds, axis=0)

## SVD
U, S, V_t = np.linalg.svd(centered, full_matrices=False)

## compute selected dimension
variance_ratio = S ** 2 / np.sum(S ** 2)
cumsum = np.cumsum(variance_ratio)
d = np.argmax(cumsum >= percentage) + 1

## write result
f.write("##########  STEP 1  ##########\n")
f.write(f"Input Percentage: {percentage}\n")
f.write(f"Selected Dimension: {d}\n")

# diagnolize single value vector for computation
S = np.diag(S)

###############################################
############# STEP 2 ##########################
###############################################


## Reconstruction
reconstructed_train_ds = (U[:,:d] @ S[0:d,:d] @ V_t[:d,:])
reconstructed_train_ds = (reconstructed_train_ds + np.mean(train_ds, axis=0)).astype(np.float64)

## write results and save reconstructed images

f.write("\n##########  STEP 2  ##########\n")
f.write("Reconstruction error\n")
mse_list = []
for i in range(len(reconstructed_train_ds)):
    reconstructed = reconstructed_train_ds[i].reshape(192,168)
    cv2.imwrite("2016112083/face%.2d.pgm" % (i + 1), reconstructed.astype(np.uint8))

    image_original = train_ds[i].reshape(192, 168)

    mse = np.mean((image_original - reconstructed) ** 2)
    mse_list.append(mse)


f.write(f"average : {np.mean(mse_list): .4f}\n")
for i in range(len(reconstructed_train_ds)):
    f.write(f"{str(i+1).zfill(2)}: {mse_list[i]: .4f}\n")


###############################################
############# STEP 3 ##########################
###############################################

f.write("\n##########  STEP 3  ##########\n")

# SVD test dataset
test_ds = np.array([cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob("faces_test/*.pgm"))])
test_ds = test_ds.reshape(-1, 192 * 168)
centered = test_ds - np.mean(test_ds,axis=0)
U, S, V_t = np.linalg.svd(centered, full_matrices=False)

variance_ratio = S ** 2 / np.sum(S ** 2)
cumsum = np.cumsum(variance_ratio)

d = np.argmax(cumsum >= percentage) + 1

S = np.diag(S)

# Reconstruct test data set
reconstructed_test_ds = (U[:,:d] @ S[0:d,:d] @ V_t[:d,:])
reconstructed_test_ds = (reconstructed_test_ds + np.mean(test_ds,axis=0)).astype(np.float64)





# Compare the Result by L2 Distance and write results
result =[]
for i in range(len(test_ds)):
    m = 987654321
    for j in range(len(reconstructed_train_ds)):

        # get image from datasets and centralize for comparing Euclidean distance
        train = reconstructed_train_ds[j] - np.mean(reconstructed_train_ds[j])
        test = reconstructed_test_ds[i] - np.mean(reconstructed_test_ds[i])
        dist = np.linalg.norm(train.astype(np.float64) - test.astype(np.float64))

        if dist < m:
            m = dist
            matched_image_index = j
    # Write Results
    f.write(f"test{str(i+1).zfill(2)}.pgm ==> face{str(matched_image_index+1).zfill(2)}.pgm\n")
    result.append(matched_image_index)
