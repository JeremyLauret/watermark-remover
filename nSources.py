# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:41:12 2018

@author: Luca
"""

import numpy as np

import matplotlib.pyplot as plt

import scipy.ndimage as ndimage

plt.rcParams['image.cmap'] = 'gray' 

""" Constants """

IMG_DIR='img/'

LISTE_NOMS=['watermarked-lena.png', 'watermarked-barbara.png', 'barbara.png']

NB_ITER = 1000

PAS_AFFICHAGE = NB_ITER//5 # Nombre d'itérations séparant deux affichages

MU = 0.01

LAMBDA = 1

"""           """

for i in range(len(LISTE_NOMS)):
    LISTE_NOMS[i] = IMG_DIR + LISTE_NOMS[i]

def show_img(img_list, nb_fig, title):
    n = len(img_list)
    plt.figure(nb_fig)
    for i in range(n):
        plt.subplot(n/3 if n/3 == n//3 else n//3, 3, i+1)
        plt.title(title + str(i))
        plt.imshow(img_list[i])
    plt.show()
    return(nb_fig + 1)

def compute_gradient(B,y,x,lam1):
    m_y=np.zeros((len(y),6))
    for i in range(len(y)):
        m_y[i][0] = np.mean(y[i])
        m_y[i][1] = np.mean(y[i]**2)    
        m_y[i][2] = np.mean(y[i]**3)    
        m_y[i][3] = np.mean(y[i]**4)    
        m_y[i][4] = np.mean(y[i]**5)    
        m_y[i][5] = np.mean(y[i]**6)
        
    K=np.zeros((len(y),4)) #base des polynomes de degré 3 pour approcher fonction score
        
    for i in range(len(y)):
        K[i][0]=1
        K[i][1]=m_y[i][0]
        K[i][2]=m_y[i][1]
        K[i][3]=m_y[i][2]
        
    M=np.zeros((len(y),4,4))
        
    for i in range(len(y)):
        M[i][0][0]=1
        M[i][0][1]=m_y[i][0]
        M[i][0][2]=m_y[i][1]
        M[i][0][3]=m_y[i][2]
        M[i][1][0]=m_y[i][0]
        M[i][1][1]=m_y[i][1]
        M[i][1][2]=m_y[i][2]
        M[i][1][3]=m_y[i][3]
        M[i][2][0]=m_y[i][1]
        M[i][2][1]=m_y[i][2]
        M[i][2][2]=m_y[i][3]
        M[i][2][3]=m_y[i][4]
        M[i][3][0]=m_y[i][2]
        M[i][3][1]=m_y[i][3]
        M[i][3][2]=m_y[i][4]
        M[i][3][3]=m_y[i][5]
        
    P=[]
    for i in range(len(y)):
        P.append([[0, 1, 2*m_y[i][0], 3*(m_y[i][1])]])
    P=np.array(P)

    w=[]
    for i in range(len(y)):
        w.append(np.linalg.inv(np.array(M[i])) @ np.array(P[i]).T)

    w=np.array(w)
    
    Psi_y=[]
    for i in range(len(y)):
        Psi_y.append(w[i][0]+w[i][1]*y[i]+w[i][2]*y[i]**2+w[i][3]*y[i]**3)
        
    M_Psi=np.zeros((len(y), len(y)))
    for i in range(len(y)):
        for j in range(len(y)):
            M_Psi[i][j]=np.mean(Psi_y[i]*x[j])
            
    for i in range(len(y)):
        y[i]=y[i]-np.mean(y[i])
    
    temp=[]
    for i in range(len(y)):
        temp.append(4 * (np.mean(y[i]**2) - 1) * y[i])

    pen=np.zeros((len(y),len(y)))
    for i in range(len(y)):
        for j in range(len(y)):
            pen[i][j]=np.mean(temp[i] * x[j])
    
    
    return(M_Psi @ B.T - np.eye(len(y)) + lam1 * pen @ B.T)


def genere_images(vecteur_noms):

    #***** lecture de l'image par exemple: xxx.jpg
    n=len(vecteur_noms)
    images_source=[]
    for i in range(n):
        temporaire=(plt.imread(vecteur_noms[i])) / 255
        if len(temporaire.shape)>2:
            images_source.append(temporaire[:,:,2])
        else:
            images_source.append(temporaire)
 
    ##****** dimensions de l'image (nb_lign,nb_col,3)

    [nb_lign, nb_col] = images_source[0].shape
      
      

    R=[]
    for i in range(n):
        R.append(images_source[i])
    #***** deconstruction de l'image a partir des composantes en des vecteurs
    R=np.array(R)
    R_L=[]
    for k in range(n):
        R_L_k=[]
        for j in range(nb_lign):
            for i in range(nb_col):
                R_L_k.append(R[k,j,i])  
    
        #**** conversion double precision pour la regle de trois Fs double()
        R_L_k = np.array(R_L_k)
        #***************** regle de trois pour se ramener dans l'intervalle [0,1]
        R_L.append(R_L_k)
    R_L=np.array(R_L)
     
    return [R_L, nb_lign, nb_col, images_source]

    
def list_to_matrix(img_list, nb_row, nb_col):
    n = len(img_list)

    img_matrix = []

    for k in range(n):
        img = np.zeros((nb_row, nb_col))

        for j in range(nb_row):
            for i in range(nb_col):
                img[i,j] = img_list[k][j+i*nb_col]

        img_matrix.append(img)

    return img_matrix

def separate_mixed(mixed_img_array, nb_iter):
    """
     * Args :
         - mixed_img_array -> tableau des observées au format liste 1D
         - nb_iter -> nombre d'itérations de descente du gradient
         
     * Returns :
         - y -> tableau des approximations au format liste 1D
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
        if (k % (nb_iter//10) == 0):
            print("Progress : ", np.floor(k / nb_iter * 100), "%")

        grad_J = compute_gradient(B, y, x, LAMBDA)

        B = B - MU * grad_J

        y = np.dot(B, x)
        
        for i in range(n):
            y[i] = y[i] - np.mean(y[i])

    return y

## Programme principal

print("Génération des images mélangées puis concaténation en vecteurs...")

[mixed_img_array, nb_lign, nb_col, images_source] = genere_images(LISTE_NOMS)

print("Recomposition des sources à partir des observées...")

y = separate_mixed(mixed_img_array, NB_ITER)

print("Recomposition terminée !")

# Affichage final

clean_img_array = []

for k in range(len(mixed_img_array)):
    clean_img_array.append((y[k] - min(y[k])) / (max(y[k]) - min(y[k])) * 255)

recomposed_img = list_to_matrix(clean_img_array, nb_lign, nb_col);

show_img(recomposed_img, 1, "Recomposition ")