## Fichier de séparation de n images mixées
import numpy as np


def compute_gradient(B, y, x, lam1):
    n = len(y)

    m_y = np.zeros((n, 6))
    for i in range(n):
        m_y[i][0] = np.mean(y[i])
        m_y[i][1] = np.mean(y[i] ** 2)
        m_y[i][2] = np.mean(y[i] ** 3)
        m_y[i][3] = np.mean(y[i] ** 4)
        m_y[i][4] = np.mean(y[i] ** 5)
        m_y[i][5] = np.mean(y[i] ** 6)

    K = np.zeros((n, 4))  # base des polynomes de degré 3 pour approcher fonction score

    for i in range(n):
        K[i][0] = 1
        K[i][1] = m_y[i][0]
        K[i][2] = m_y[i][1]
        K[i][3] = m_y[i][2]

    M = np.zeros((n, 4, 4))

    for i in range(n):
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
    for i in range(n):
        P.append([[0, 1, 2 * m_y[i][0], 3 * (m_y[i][1])]])
    P = np.array(P)

    w = []
    for i in range(n):
        w.append(np.linalg.inv(np.array(M[i])) @ np.array(P[i]).T)

    w = np.array(w)

    Psi_y = []
    for i in range(n):
        Psi_y.append(w[i][0] + w[i][1] * y[i] + w[i][2] * y[i] ** 2 + w[i][3] * y[i] ** 3)

    M_Psi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M_Psi[i][j] = np.mean(Psi_y[i] * x[j])

    for i in range(n):
        y[i] = y[i] - np.mean(y[i])

    temp = []
    for i in range(n):
        temp.append(4 * (np.mean(y[i] ** 2) - 1) * y[i])

    pen = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pen[i][j] = np.mean(temp[i] * x[j])

    return (M_Psi @ B.T - np.eye(n) + lam1 * pen @ B.T)

def compute_gradient(B, y, x, lam1):
    n = len(y)

    m_y = np.zeros((n, 6))
    for i in range(n):
        m_y[i][0] = np.mean(y[i])
        m_y[i][1] = np.mean(y[i] ** 2)
        m_y[i][2] = np.mean(y[i] ** 3)
        m_y[i][3] = np.mean(y[i] ** 4)
        m_y[i][4] = np.mean(y[i] ** 5)
        m_y[i][5] = np.mean(y[i] ** 6)

    K = np.zeros((n, 4))  # base des polynomes de degré 3 pour approcher fonction score

    for i in range(n):
        K[i][0] = 1
        K[i][1] = m_y[i][0]
        K[i][2] = m_y[i][1]
        K[i][3] = m_y[i][2]

    M = np.zeros((n, 4, 4))

    for i in range(n):
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
    for i in range(n):
        P.append([[0, 1, 2 * m_y[i][0], 3 * (m_y[i][1])]])
    P = np.array(P)

    w = []
    for i in range(n):
        w.append(np.linalg.inv(np.array(M[i])) @ np.array(P[i]).T)

    w = np.array(w)

    Psi_y = []
    for i in range(n):
        Psi_y.append(w[i][0] + w[i][1] * y[i] + w[i][2] * y[i] ** 2 + w[i][3] * y[i] ** 3)

    M_Psi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M_Psi[i][j] = np.mean(Psi_y[i] * x[j])

    for i in range(n):
        y[i] = y[i] - np.mean(y[i])

    temp = []
    for i in range(n):
        temp.append(4 * (np.mean(y[i] ** 2) - 1) * y[i])

    pen = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pen[i][j] = np.mean(temp[i] * x[j])

    return (M_Psi @ B.T - np.eye(n) + lam1 * pen @ B.T)

def separate_mixed_unicolor(input_list_v, nb_iter, lambd, mu):
    """
     * Args :
         - input_list_v -> liste des observées (vecteurs)
         - nb_iter -> nombre d'itérations de descente du gradient
         - lambd, mu

     * Returns :
         - output_list_v -> liste des approximations (vecteurs)
         - dtype_list -> type de données des observées
    """
    n = len(input_list_v)
    B = np.eye(n)
    dtype_list = []

    for i in range(n):
        dtype_list.append(input_list_v[i].dtype)

    ## Normalisation
    for i in range(n):
        input_list_v[i] = (input_list_v[i] - np.mean(input_list_v[i])) / np.std(input_list_v[i])

    output_list_v = input_list_v

    for k in range(nb_iter):
        if (k % (nb_iter // 10 if nb_iter >= 10 else nb_iter) == 0):
            print("Progress : ", np.floor(k / nb_iter * 100), "%")

        grad_J = compute_gradient(B, output_list_v, input_list_v, lambd)

        B = B - mu * grad_J

        output_list_v = np.dot(B, input_list_v)

        for i in range(n):
            output_list_v[i] = output_list_v[i] - np.mean(output_list_v[i])

    return output_list_v, dtype_list


def separate_mixed_color(input_list_v, nb_iter, lambd, mu):
    """
     * Args :
         - input_list_v -> tableau des observées en couleur (vecteurs x 3)
         - nb_iter -> nombre d'itérations de descente du gradient

     * Returns :
         - output_list_v -> tableau des approximations en couleur (vecteurs x 3)
         - dtype_list
    """
    n = len(input_list_v)
    col_number = input_list_v[0].shape[1]

    input_list_v_col = [[] for i in range(col_number)]
    output_list_v_col = []
    output_list_v = []

    for i in range(col_number):
        for j in range(n):
            input_list_v_col[i].append(input_list_v[j][:, i])

    for i in range(col_number):
        output_v_col, dtype_list = separate_mixed_unicolor(input_list_v_col[i], nb_iter, lambd, mu)
        output_list_v_col.append(output_v_col)

    for i in range(n):
        output_v = np.zeros(
            (output_list_v_col[0][0].shape[0], col_number),
            dtype=np.float_
        )

        for j in range(col_number):
            output_v[:, j] = output_list_v_col[j][i]

        output_list_v.append(output_v)

    return output_list_v, dtype_list


def separate_mixed(input_list_v, nb_iter, lambd, mu):
    """
     * Args :
         - input_list_v -> liste des observées (vecteurs [x 3])
         - nb_iter -> nombre d'itérations de descente du gradient

     * Returns :
         - output_list_v -> liste des approximations (vecteurs [x 3])
         - dtype_list -> type de données des observées
    """
    if (len(input_list_v[0].shape) > 1):  # Images en couleur
        output_list_v, dtype_list = separate_mixed_color(input_list_v, nb_iter, lambd, mu)

    else:
        output_list_v, dtype_list = separate_mixed_unicolor(input_list_v, nb_iter, lambd, mu)

    return output_list_v, dtype_list


# Cette fonction n'a pas abouti et n'est probablement plus compatible avec le reste du programme


def separate_sub_mixed(mixed_sub_img_vect): # Work in progress
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