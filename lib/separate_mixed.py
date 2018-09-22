## Fichier de séparation de n images mixées


def compute_gradient(B, y, x, lam1):
    m_y = np.zeros((len(y), 6))
    for i in range(len(y)):
        m_y[i][0] = np.mean(y[i])
        m_y[i][1] = np.mean(y[i] ** 2)
        m_y[i][2] = np.mean(y[i] ** 3)
        m_y[i][3] = np.mean(y[i] ** 4)
        m_y[i][4] = np.mean(y[i] ** 5)
        m_y[i][5] = np.mean(y[i] ** 6)

    K = np.zeros((len(y), 4))  # base des polynomes de degré 3 pour approcher fonction score

    for i in range(len(y)):
        K[i][0] = 1
        K[i][1] = m_y[i][0]
        K[i][2] = m_y[i][1]
        K[i][3] = m_y[i][2]

    M = np.zeros((len(y), 4, 4))

    for i in range(len(y)):
        M[i][0][0] = 1
        M[i][0][1] = m_y[i][0]
        M[i][0][2] = m_y[i][1]
        M[i][0][3] = m_y[i][2]
        M[i][1][0] = m_y[i][0]
        M[i][1][1] = m_y[i][1]
        M[i][1][2] = m_y[i][2]
        M[i][1][3] = m_y[i][3]
        M[i][2][0] = m_y[i][1]
        M[i][2][1] = m_y[i][2]
        M[i][2][2] = m_y[i][3]
        M[i][2][3] = m_y[i][4]
        M[i][3][0] = m_y[i][2]
        M[i][3][1] = m_y[i][3]
        M[i][3][2] = m_y[i][4]
        M[i][3][3] = m_y[i][5]

    P = []
    for i in range(len(y)):
        P.append([[0, 1, 2 * m_y[i][0], 3 * (m_y[i][1])]])
    P = np.array(P)

    w = []
    for i in range(len(y)):
        w.append(np.linalg.inv(np.array(M[i])) @ np.array(P[i]).T)

    w = np.array(w)

    Psi_y = []
    for i in range(len(y)):
        Psi_y.append(w[i][0] + w[i][1] * y[i] + w[i][2] * y[i] ** 2 + w[i][3] * y[i] ** 3)

    M_Psi = np.zeros((len(y), len(y)))
    for i in range(len(y)):
        for j in range(len(y)):
            M_Psi[i][j] = np.mean(Psi_y[i] * x[j])

    for i in range(len(y)):
        y[i] = y[i] - np.mean(y[i])

    temp = []
    for i in range(len(y)):
        temp.append(4 * (np.mean(y[i] ** 2) - 1) * y[i])

    pen = np.zeros((len(y), len(y)))
    for i in range(len(y)):
        for j in range(len(y)):
            pen[i][j] = np.mean(temp[i] * x[j])

    return (M_Psi @ B.T - np.eye(len(y)) + lam1 * pen @ B.T)

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