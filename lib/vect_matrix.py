## Fichier d'outils de convertion de matrices en vecteurs et vice-versa
import numpy as np


def matrix_to_vect(img_matrix): #
    """
     * Args :
         - img_matrix -> tableau de dimensions nb_row x nb_col représentant une image

     * Returns :
         - img_vect -> vecteur 1D des lignes de la matrice mises bout à bout
         - nb_row, nb_col
    """
    nb_row, nb_col = img_matrix.shape[0:2]

    img_vect = np.zeros((nb_row*nb_col))

    for i in range(nb_row):
        for j in range(nb_col):
            img_vect[nb_col*i + j] = img_matrix[i, j]

    return img_vect, nb_row, nb_col


def vect_to_matrix(img_vect, nb_row, nb_col): #
    """
     * Args :
         - img_vect -> vecteur 1D des lignes d'une images mises bout à bout
         - nb_row, nb_col -> dimensions de la matrice retournée

     * Returns :
         - img_matrix -> tableau de dimensions nb_row x nb_col représentant l'image
    """
    img_matrix = np.zeros((nb_row, nb_col))

    for i in range(nb_row):
        for j in range(nb_col):
            img_matrix[i, j] = img_vect[i*nb_col + j]

    return img_matrix


def matrix_to_vect_array(img_matrix_array): #
    """
     * Args :
         - img_matrix_array -> tableau des images (matrices nb_row x nb_col [x 3])

     * Returns :
         - img_vect_array -> tableau des images (vecteurs [x 3])
         - nb_row, nb_col
    """
    n = len(img_matrix_array)

    nb_row, nb_col = img_matrix_array[0].shape[0:2]

    if (len(img_matrix_array[0].shape) > 2):  # Images colorées
        img_vect_array = [np.zeros((nb_row*nb_col, img_matrix_array[k].shape[2])) for k in range(n)]

        for i in range(n):
            for j in range(img_matrix_array[0].shape[2]):
                img_vect_array[i][:, j] += matrix_to_vect(img_matrix_array[i][:, :, j])[0]

    else:
        img_vect_array = [np.zeros(nb_row*nb_col) for k in range(n)]

        for i in range(n):
            img_vect_array[i] += matrix_to_vect(img_matrix_array[i])[0]

    return img_vect_array, nb_row, nb_col


def vect_to_matrix_array(img_vect_array, nb_row, nb_col): #
    """
     * Args :
         - img_vect_array -> tableau des images (vecteurs [x 3])
         - nb_row, nb_col -> dimensions des matrices retournées

     * Returns :
         - img_matrix_array -> tableau des images (matrices nb_row x nb_col [x 3])
    """
    n = len(img_vect_array)

    if (len(img_vect_array[0].shape) > 1):  # Images colorées
        img_matrix_array = [np.zeros((nb_row, nb_col, img_vect_array[k].shape[1])) for k in range(n)]

        for i in range(n):
            for j in range(img_vect_array[0].shape[1]):
                img_matrix_array[i][:, :, j] += vect_to_matrix(img_vect_array[i][:, j], nb_row, nb_col)

    else:
        img_matrix_array = [np.zeros((nb_row, nb_col)) for k in range(n)]

        for i in range(n):
            img_matrix_array[i] += vect_to_matrix(img_vect_array[i], nb_row, nb_col)

    return img_matrix_array


def sub_matrix_to_sub_vector(mixed_sub_img_matrix):
    """
     * Args :
         - mixed_sub_img_matrix -> matrice de tableaux contenant des sous-images (matrices [x 3])

     * Returns :
         - mixed_sub_img_vect -> matrice de tableaux contenant des sous-images (vecteurs [x 3])
         - nb_row_tiles, nb_col_tiles -> dimensions des sous-images sous forme de matrices
    """
    nb_row_tiles, nb_col_tiles = mixed_sub_img_matrix[0, 0].shape[0:2]

    nb_row, nb_col = mixed_sub_img_matrix.shape[0:2]

    mixed_sub_img_vect = np.zeros(nb_row, nb_col)

    for i in range(nb_row):
        for j in range(nb_col):
            mixed_sub_img_vect[i, j] = matrix_to_vect_array(mixed_sub_img_matrix[i, j])[0]

    return mixed_sub_img_vect, nb_row_tiles, nb_col_tiles


def sub_vector_to_sub_matrix(sub_img_vect, nb_row_tiles, nb_col_tiles):
    """
     * Args :
         - sub_image_vect -> matrice de tableaux contenant les approximées des sous-images (vecteurs [x 3])
         - nb_row_tiles, nb_col_tiles -> dimensions des sous-images sous forme de matrices

     * Returns :
         - sub_image_matrix -> matrice de tableaux contenant les approximées des sous-images (matrices [x 3])
    """
    nb_row, nb_col = mixed_sub_img_vect.shape[0:2]

    sub_img_matrix = np.zeros(nb_row, nb_col)

    for i in range(nb_row):
        for j in range(nb_col):
            sub_img_matrix[i, j] = vect_to_matrix_array(sub_img_vect[i, j], nb_row_tiles, nb_col_tiles)

    return sub_img_matrix