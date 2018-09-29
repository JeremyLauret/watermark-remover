## Fichier d'outils de convertion de matrices en vecteurs et vice-versa
import numpy as np


def matrix_to_vect(matrix):
    """
     * Only works for dim 2 or dim 3 matrix

     * Args :
         - img_matrix -> tableau de dimensions nb_row x nb_col [x 3 | x 4] représentant une image

     * Returns :
         - vect -> vecteur 1D des lignes de la matrice mises bout à bout
         - nb_row, nb_col
    """
    nb_row, nb_col = matrix.shape[0:2]

    vect = np.zeros(tuple([nb_row*nb_col]) + matrix.shape[2:], dtype=matrix.dtype)

    if (len(matrix.shape) == 2):
        for i in range(nb_row):
            for j in range(nb_col):
                vect[nb_col*i + j] = matrix[i, j]

    elif (len(matrix.shape) == 3):
        for i in range(nb_row):
            for j in range(nb_col):
                vect[nb_col*i + j, :] = matrix[i, j, :]

    else:
        print('Erreur : longueur de la matrice incompatible avec la conversion en vecteur.')

    return vect, nb_row, nb_col


def vect_to_matrix(vect, nb_row, nb_col):
    """
     * Only works for dim 1 or dim 2 vect

     * Args :
         - vect -> vecteur 1D des lignes d'une images mises bout à bout
         - nb_row, nb_col -> dimensions de la matrice retournée

     * Returns :
         - matrix -> tableau de dimensions nb_row x nb_col représentant l'image
    """
    matrix = np.zeros((nb_row, nb_col) + vect.shape[1:], dtype=vect.dtype)

    if (len(vect.shape) == 1):
        for i in range(nb_row):
            for j in range(nb_col):
                matrix[i, j] = vect[i*nb_col + j]

    elif (len(vect.shape) == 2):
        for i in range(nb_row):
            for j in range(nb_col):
                matrix[i, j, :] = vect[i*nb_col + j, :]

    return matrix


def matrix_to_vect_list(matrix_list):
    """
     * Only works for dim 2 or dim 3 matrix

     * Args :
         - matrix_list -> liste des images (matrices nb_row x nb_col [x 3 | x 4])

     * Returns :
         - vect_list -> liste des images (vecteurs nb_row*nb_col [x 3 | x 4])
         - nb_row_list, nb_col_list
    """
    n = len(matrix_list)
    nb_row_list, nb_col_list, vect_list = [], [], []

    for i in range(n):
        converted = matrix_to_vect(matrix_list[i])
        vect_list.append(converted[0])
        nb_row_list.append(converted[1])
        nb_col_list.append(converted[2])

    return vect_list, nb_row_list, nb_col_list


def vect_to_matrix_list(vect_list, nb_row_list, nb_col_list):
    """
     * Only works for dim 1 or dim 2 vect

     * Args :
         - vect_list -> liste des images (vecteurs nb_row*nb_col [x 3 | x 4])
         - nb_row_list, nb_col_list -> listes des dimensions des matrices retournées

     * Returns :
         - matrix_list -> liste des images (matrices nb_row x nb_col [x 3 | x 4])
    """
    n = len(vect_list)
    matrix_list = []

    for i in range(n):
        matrix_list.append(vect_to_matrix(vect_list[i], nb_row_list[i], nb_col_list[i]))

    return matrix_list


# Ces fonctions n'ont pas abouti et ne sont probablement plus compatibles avec le reste du programme


def sub_matrix_to_sub_vector(mixed_sub_img_matrix): # Work in progress
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


def sub_vector_to_sub_matrix(sub_img_vect, nb_row_tiles, nb_col_tiles): # Work in progress
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