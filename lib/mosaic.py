def mosaique(liste_images, cote):
    (hauteur, largeur) = liste_images[0].shape[0:2]
    for I in range(hauteur // cote):
        for J in range(largeur // cote):
            carreaux = []
            for k in range(len(liste_images)):
                carreaux.append(liste_images[k][I * cote:(I + 1) * cote, J * cote:(J + 1) * cote])
            carreaux, nb_row, nb_col = matrix_to_vect_array(carreaux)
            carreaux = separate_mixed(carreaux, NB_ITER)
            carreaux = vect_to_matrix_array(carreaux, nb_row, nb_col)
            for k in range(len(liste_images)):
                liste_images[k][I * cote:(I + 1) * cote, J * cote:(J + 1) * cote] = carreaux[k]
    return liste_images


def create_sub_matrix_array(mixed_img_matrix_array, square_side, method='pixel_pick'): # Work in progress
    """
     * Args :
         - mixed_img_matrix_array -> tableau des observées (matrices [x 3])
         - square_side -> longueur du côté des carreaux servant à créer les sous-images
         - method = 'pixel_pick' -> méthode de création des sous-images

     * Returns :
         - mixed_sub_img_matrix -> matrice de tableaux contenant des sous-images (matrices [x 3])
    """
    n = len(mixed_img_matrix_array)

    nb_row, nb_col = mixed_img_matrix_array[0].shape[0:2]

    if (method == 'pixel_pick'):
        mixed_sub_img_matrix = np.zeros((square_side, square_side, n, nb_row // square_side, nb_col // square_side))
        for i in range(square_side):
            for j in range(square_side):
                tile = np.zeros((n, nb_row // square_side, nb_col // square_side))

                for k in range(n):
                    tile[k] = mixed_img_matrix_array[k][i: nb_row: square_side, j: nb_col: square_side]

                mixed_sub_img_matrix[i, j] = tile

    elif (method == 'mosaic'):
        mixed_sub_img_matrix = np.zeros((nb_row // square_side, nb_col // square_side, n, square_side, square_side))

        for i in range(nb_row // square_side):
            for j in range(nb_col // square_side):
                tile = np.zeros((n, square_side, square_side))
                for k in range(n):
                    tile[k] = mixed_img_matrix_array[k][i * square_side: (i + 1) * square_side,
                              j * square_side: (j + 1) * square_side]

                mixed_sub_img_matrix[i, j] = tile

    return mixed_sub_img_matrix


def revert_sub_matrix_array(sub_img_matrix, method='pixel_pick'): # Work in progress
    """
     * Args :
         - sub_image_matrix -> matrice de tableaux contenant les approximées des sous-images (matrices [x 3])
         - method = 'pixel_pick' -> méthode utilisée pour la création des sous-images

     * Returns :
         - y -> tableau des approximations (matrices [x 3])
    """
    n = len(sub_img_matrix[0, 0])

    if (method == 'pixel_pick'):
        square_side = sub_img_matrix.shape[0]

        nb_row, nb_col = [x * square_side for x in sub_img_matrix[0, 0, 0].shape[0:2]]

        y = np.zeros((n, nb_row, nb_col))

        for k in range(n):
            y[k] = np.zeros((nb_row, nb_col))

            for i in range(nb_row):
                for j in range(nb_col):
                    y[k, i, j] = sub_img_matrix[i % square_side, j % square_side, k, i // square_side, j // square_side]

    elif (method == 'mosaic'):
        square_side = sub_img_matrix[0, 0, 0].shape[0]

        nb_row, nb_col = [x * square_side for x in sub_img_matrix.shape[0:2]]

        y = np.zeros(n)

        for k in range(n):
            y[k] = np.zeros(nb_row, nb_col)

            for i in range(nb_row):
                for j in range(nb_col):
                    y[k, i, j] = sub_img_matrix[i // square_side, j // square_side, k, i % square_side, j % square_side]

    return y


def mosaique2(liste_images, cote):
    (hauteur, largeur) = liste_images[0].shape[0:2]
    liste_carreaux=[]
    for i in range(cote):
        for j in range(cote):
            carreaux = []
            for k in range(len(liste_images)):
                sous_image=np.zeros((hauteur // cote, largeur // cote))
                for I in range(hauteur // cote):
                    for J in range(largeur // cote):
                        sous_image[I,J]=liste_images[k][I*cote+i, J*cote+j]
                carreaux.append(sous_image)
            carreaux, nb_row, nb_col = matrix_to_vect_array(carreaux)
            carreaux = separate_mixed(carreaux, NB_ITER)
            carreaux = vect_to_matrix_array(carreaux, nb_row, nb_col)
            liste_carreaux.append(carreaux)
            for k in range(len(liste_images)):
                for I in range(hauteur // cote):
                    for J in range(largeur // cote):
                        liste_images[k][I*cote+i, J*cote+j] = liste_carreaux[i*(cote)+j][k][I,J]
    return liste_images