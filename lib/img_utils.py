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


def load_img_from_name(names_list): #
    n = len(names_list)
    matrix_list = []

    for i in range(n):
        matrix_list.append(plt.imread(names_list[i]))

    return(matrix_list)


def color_to_gray(colored_matrix): #
    gray_matrix = np.zeros((colored_matrix.shape[0:2]))

    if (len(colored_matrix.shape) > 2):
        gray_matrix += (colored_matrix[:,:,0] + colored_matrix[:,:,1] + colored_matrix[:,:,2]) / 3
    else:
        gray_matrix += colored_matrix

    return gray_matrix


def gray_in_list(matrix_list): #
    counter = 0
    gray_needed = False

    while (counter < len(matrix_list) and gray_needed == False) :
        if (len(matrix_list[counter].shape) < 2) :
            gray_needed = True
        counter += 1

    return(gray_needed)


def unnormalize(normalized_array):
    unnormalized_array = []

    for k in range(len(normalized_array)):
        unnormalized_array.append((normalized_array[k] - min(normalized_array[k]))
                                  / (max(normalized_array[k]) - min(normalized_array[k])) * 255)

    return(unnormalized_array)


def mix_img(img_matrix_1, img_matrix_2, alpha, output_name) :
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