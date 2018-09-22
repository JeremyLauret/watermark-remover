## Fichier de calcul du gradient


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