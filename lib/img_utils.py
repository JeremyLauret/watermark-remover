## Fichier d'outils de manipulation des images
import matplotlib.pyplot as plt
import numpy as np

def show_img(img_list, nb_fig, title): #
    """
     * Shows the matrix images contained in img_list
    """
    n = len(img_list)
    plt.figure(nb_fig)
    for i in range(n):
        plt.subplot(np.ceil(n/3), 3 if n >= 3 else n, i+1)
        plt.title(title + str(i))
        plt.imshow(img_list[i])
    plt.show()
    return(nb_fig + 1)

def del_opacity(matrix):
    if(len(matrix.shape) == 3 and matrix.shape[2] == 4):
        return matrix[:,:,0:3]

    return matrix

def load_img_from_name(names_list): #
    n = len(names_list)
    matrix_list = []

    for i in range(n):
        matrix_list.append(del_opacity(plt.imread(names_list[i])))

    return(matrix_list)


def color_to_gray(matrix):
    if (len(matrix.shape) <= 2) :
        return matrix

    else :
        gray_matrix = np.zeros(matrix.shape[0:2], dtype=matrix.dtype)
        # Thanks Wikipedia : https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
        gray_matrix += matrix.dtype.type(0.2126*matrix[:, :, 0] + 0.7152*matrix[:, :, 1] + 0.0722*matrix[:, :, 2])
        return gray_matrix


def gray_in_list(matrix_list): #
    counter = 0
    gray_needed = False

    while (counter < len(matrix_list) and gray_needed == False) :
        if (len(matrix_list[counter].shape) <= 2) :
            gray_needed = True
        counter += 1

    return gray_needed


def rev_norm(list):
    """
     * Reverts normalization in the given list
    """
    rev_list = []

    for k in range(len(list)):
        rev_list.append((list[k]-np.min(list[k])) / (np.max(list[k])-np.min(list[k])) * 255)

    return rev_list


def mix_img(img_matrix_1, img_matrix_2, alpha, output_name) : # Work in progress
    """
     * Args :
         - img_matrix_1, img_matrix_2 -> images à mélanger (matrices)
         - alpha -> proportions du mélange (alpha de l'image 1, (1-alpha) de l'image 2)
         - output_name -> nom de l'image sauvegardée
    """
    if (img_matrix_1.shape[0:2] != img_matrix_2.shape[0:2]) :
        print("Erreur : images de tailles différentes!")
        return

    if (len(img_matrix_1.shape) == len(img_matrix_2.shape)) :
        mixed_matrix = alpha * img_matrix_1 + (1 - alpha) * img_matrix_2

    elif (len(img_matrix_1.shape) > 2) :
        mixed_matrix = np.zeros_like(img_matrix_1)
        for k in range(mixed_matrix.shape[2]) :
            mixed_matrix[:, :, k] = alpha * img_matrix_1[:, :, k] + (1 - alpha) * img_matrix_2

    else :
        mixed_matrix = np.zeros_like(img_matrix_2)
        for k in range(mixed_matrix.shape[2]) :
            mixed_matrix[:, :, k] = alpha * img_matrix_1 + (1 - alpha) * img_matrix_2[:, :, k]

    plt.imsave(output_name, mixed_matrix)
    return