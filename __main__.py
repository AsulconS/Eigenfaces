from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os


dataset_path = 'Dataset/'
dataset_dir  = os.listdir(dataset_path)

train_image_names = ['subject01.normal.jpg', 'subject02.normal.jpg', 'subject03.normal.jpg', 'subject07.normal.jpg', 'subject10.normal.jpg', 'subject11.normal.jpg', 'subject14.normal.jpg', 'subject15.normal.jpg']

width  = 195
height = 231
s = width * height
tensor_train = np.ndarray(shape=(8, s), dtype=np.float64)
tensor_tests = np.ndarray(shape=(17, s), dtype=np.float64)
norm_tensor_train = np.ndarray(shape=(8, s), dtype=np.float64)


def load_training_images():
    for i in range(8):
        img = plt.imread(dataset_path + train_image_names[i])
        tensor_train[i,:] = np.array(img, dtype=np.float64).flatten()
        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')
    plt.suptitle("Train Faces")
    plt.show()


def load_testing_images():
    global test_image_names
    test_image_names = dataset_dir
    for i in range(17):
        img = plt.imread(dataset_path + test_image_names[i])
        tensor_tests[i,:] = np.array(img, dtype=np.float64).flatten()
        plt.subplot(3, 6, i + 1)
        plt.title(test_image_names[i].split('.')[0][-2:] + test_image_names[i].split('.')[1])
        plt.imshow(img, cmap='gray')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')
    plt.suptitle("Test Faces")
    plt.show()


def calculate_mean_face():
    global mean_face
    mean_face = np.zeros((1, s))
    for i in tensor_train:
        mean_face = np.add(mean_face, i)

    mean_face = np.divide(mean_face, 8.0).flatten()
    plt.imshow(mean_face.reshape(height, width), cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')
    plt.suptitle("Mean Face")
    plt.show()


def calculate_norm_faces():
    for i in range(8):
        norm_tensor_train[i] = np.subtract(tensor_train[i], mean_face)
        img = norm_tensor_train[i].reshape(height, width)
        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')
    plt.suptitle("Normalized Test Faces")
    plt.show()


def calculate_cov_matrix():
    global cov_matrix
    cov_matrix = np.cov(norm_tensor_train)
    cov_matrix = np.divide(cov_matrix, 8.0)
    print(f'X Covariance Matrix:\n{cov_matrix}')


def find_eig():
    global sorted_eigenvalues, sorted_eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print(f'Eigenvalues:\n{eigenvalues}')
    print(f'Eigenvectors:\n{eigenvectors}')
    pairs = [(eigenvalues[i], eigenvectors[:,i]) for i in range(len(eigenvalues))]
    pairs.sort(reverse=True)
    sorted_eigenvalues  = [pairs[i][0] for i in range(len(eigenvalues))]
    sorted_eigenvectors = [pairs[i][1] for i in range(len(eigenvalues))]


def find_cumulative_variance():
    cum_prop_var = np.cumsum(sorted_eigenvalues) / sum(sorted_eigenvalues)
    print(f'Cumulative proportion of variance explained vector:\n{cum_prop_var}')
    num_comp = range(1, len(sorted_eigenvalues) + 1)
    plt.title('Cum. Prop. Variance Explain and Components Kept')
    plt.xlabel('Principal Components')
    plt.ylabel('Cum. Prop. Variance Expalined')
    plt.scatter(num_comp, cum_prop_var)
    plt.show()


def crop_and_project_data():
    global proj_data
    cropped_data = np.array(sorted_eigenvectors[:7]).transpose()
    proj_data = np.dot(tensor_train.transpose(), cropped_data)
    proj_data = proj_data.transpose()
    for i in range(proj_data.shape[0]):
        img = proj_data[i].reshape(height, width)
        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')
    plt.suptitle('Eigenfaces')
    plt.show()


def calculate_weights():
    global w
    w = np.array([np.dot(proj_data, i) for i in norm_tensor_train])
    print(f'Weights:\n{w}')


count = 0
num_images = 0
correct_pred = 0
def recognise(img):
    global count, highest_min, num_images, correct_pred
    unknown_face = plt.imread('Dataset/'+img)
    num_images += 1
    unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()
    normalised_uface_vector = np.subtract(unknown_face_vector, mean_face)

    plt.subplot(9, 4, count + 1)
    plt.imshow(unknown_face, cmap='gray')
    plt.title('Input:'+'.'.join(img.split('.')[:2]))
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off',left='off', which='both')
    count += 1

    w_unknown = np.dot(proj_data, normalised_uface_vector)
    diff  = w - w_unknown
    norms = np.linalg.norm(diff, axis=1)
    index = np.argmin(norms)

    t1 = 100111536
    #t1 = 200535910.268 # working with 6 faces
    #t0 = 86528212
    t0 = 88831687
    #t0 = 143559033 # working with 6 faces

    if norms[index] < t1:
        plt.subplot(9, 4, count + 1)
        if norms[index] < t0: # It's a face
            if img.split('.')[0] == train_image_names[index].split('.')[0]:
                plt.title('Matched:'+'.'.join(train_image_names[index].split('.')[:2]), color='g')
                plt.imshow(imread('Dataset/'+train_image_names[index]), cmap='gray')
                correct_pred += 1
            else:
                plt.title('Matched:'+'.'.join(train_image_names[index].split('.')[:2]), color='r')
                plt.imshow(imread('Dataset/'+train_image_names[index]), cmap='gray')
        else:
            if img.split('.')[0] not in [i.split('.')[0] for i in train_image_names] and img.split('.')[0] != 'apple':
                plt.title('Unknown face!', color='g')
                plt.imshow(imread('blank.png'))
                correct_pred += 1
            else:
                plt.title('Unknown face!', color='r')
                plt.imshow(imread('blank.png'))
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off',left='off', which='both')
        plt.subplots_adjust(right=1.2, top=2.5)
    else:
        plt.subplot(9, 4, count + 1)
        if len(img.split('.')) == 3:
            plt.title('Not a face!', color='r')
            plt.imshow(imread('blank.png'))
        else:
            plt.title('Not a face!', color='g')
            plt.imshow(imread('blank.png'))
            correct_pred += 1
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off',left='off', which='both')
    count += 1


def init():
    fig = plt.figure(figsize=(15, 15))
    recognise('apple.jpg')
    for i in range(17):
        recognise(test_image_names[i])
    plt.tight_layout()
    plt.show()
    print(f'Correct predictions: {correct_pred}/{num_images} = {correct_pred / num_images * 100.00}')


def main():
    load_training_images()
    load_testing_images()
    calculate_mean_face()
    calculate_norm_faces()
    calculate_cov_matrix()
    find_eig()
    find_cumulative_variance()
    crop_and_project_data()
    calculate_weights()
    init()


if __name__ == '__main__':
    main()
