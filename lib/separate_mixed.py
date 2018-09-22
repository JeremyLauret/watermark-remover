## Fichier de séparation de n images mixées
import gradient.compute_gradient


def separate_mixed_unicolor(mixed_img_array, nb_iter):
    """
     * Args :
         - mixed_img_array -> tableau des observées (vecteurs)
         - nb_iter -> nombre d'itérations de descente du gradient

     * Returns :
         - y -> tableau des approximations (vecteurs)
    """
    n = len(mixed_img_array)

    B = np.eye(n)

    ## Normalisation ##
    for i in range(n):
        mixed_img_array[i] = mixed_img_array[i] - np.mean(mixed_img_array[i])
        mixed_img_array[i] = mixed_img_array[i] / np.std(mixed_img_array[i])
    ##               ##

    x = mixed_img_array

    y = x

    for k in range(nb_iter):
        if (k % (nb_iter // 2) == 0):
            print("Progress : ", np.floor(k / nb_iter * 100), "%")

        grad_J = compute_gradient(B, y, x, LAMBDA)

        B = B - MU * grad_J

        y = np.dot(B, x)

        for i in range(n):
            y[i] = y[i] - np.mean(y[i])

    return y


def separate_mixed_color(mixed_img_array_color, nb_iter):
    """
     * Args :
         - mixed_img_array_color -> tableau des observées en couleur (vecteurs x 3)
         - nb_iter -> nombre d'itérations de descente du gradient

     * Returns :
         - y -> tableau des approximations en couleur (vecteurs x 3)
    """
    n = len(mixed_img_array_color)

    y = [np.zeros(mixed_img_array_color[i].shape) for i in range(n)]

    for couleur in range(mixed_img_array_color[0].shape[2]):
        colorList = []

        for i in range(n):
            colorList.append(mixed_img_array_color[i][:, :, couleur])

        colorList = separate_mixed_unicolor(colorList, nb_iter)

        for j in range(n):
            y[j][:, :, couleur] += colorList[j]

    return y


def separate_mixed(mixed_img_array, nb_iter):
    """
     * Args :
         - mixed_img_array -> tableau des observées (vecteurs [x 3])
         - nb_iter -> nombre d'itérations de descente du gradient

     * Returns :
         - y -> tableau des approximations (vecteurs [x 3])
    """
    if (len(mixed_img_array[0].shape) > 1):  # Images en couleur
        y = separate_mixed_color(mixed_img_array, nb_iter)

    else:
        y = separate_mixed_unicolor(mixed_img_array, nb_iter)

    return y


def separate_sub_mixed(mixed_sub_img_vect):
    """
     * Args :
         - mixed_sub_img_vect -> matrice de tableaux contenant des sous-images (vecteurs [x 3])

     * Returns :
         - y -> matrice de tableaux contenant les approximées des sous-images (vecteurs [x 3])
    """
    nb_row, nb_col = mixed_sub_img_vect.shape[0:2]

    y = np.zeros(mixed_sub_img_vect.shape[0:2])

    for i in range(nb_row):
        for j in range(nb_col):
            y[i, j] = separate_mixed(mixed_sub_img_vect)

    return y