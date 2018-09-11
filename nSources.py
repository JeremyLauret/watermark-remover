# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:41:12 2018

@author: Luca
"""

import numpy as np 
# this is the key library for manipulating arrays. Use the online ressources! http://www.numpy.org/

import matplotlib.pyplot as plt 
# used to read images, display and plot http://matplotlib.org/api/pyplot_api.html

import scipy.ndimage as ndimage
# one of several python libraries for image procession

plt.rcParams['image.cmap'] = 'gray' 

LISTE_NOMS=['lena_W.png', 'barbara.png', 'barbara_W.png']


def compute_gradient(B,y,x,lam1):
    m_y=np.zeros((len(y),6))
    for i in range(len(y)):
        m_y[i][0]=np.mean(y[i])
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

    
def recons_images_test_nb(vect_images_vect, nb_lign, nb_col):
    n=len(vect_images_vect)
    images_matrice=[]
    #np.zeros((n, nb_lign, nb_col))
    for k in range(n):
        reconstruction_image_matrice=np.zeros((nb_lign, nb_col))
        for j in range(nb_lign):
            #images_matrice[k][j] = vect_images_vect[k][ nb_col*j : nb_col*(j+1)]
            for i in range(nb_col):
                reconstruction_image_matrice[i,j]=vect_images_vect[k][j+i*nb_col]
        images_matrice.append(reconstruction_image_matrice)


    return images_matrice


# Programme principal
def main():
    #####
    print("Generation puis concatenation des images en vecteur .....")
    
    # s est un vecteur de vecteurs source
    [vect_vect_source, nb_lign, nb_col, images_source] = genere_images(LISTE_NOMS)
    n = len(images_source)
    
    for i in range(n):
        vect_vect_source[i]=vect_vect_source[i]-np.mean(vect_vect_source[i])
        vect_vect_source[i]=vect_vect_source[i]/np.std(vect_vect_source[i])

    plt.figure(1)
    for i in range(n):
        plt.figure(i+1)
        #numero=220+i
        #plt.subplot(i+1)
        plt.title("Source "+str(i))
        plt.imshow(images_source[i])
    
    x=vect_vect_source

    
    print('l algo tourne.......')

    nb_iter = 1001
    # Initialisation de la matrice de separation
    B = np.eye(n)
    
    # pas dans la descente du gradient
    mu=0.01
    
    # hyperparametre : parametre de penalisation : je cherche des sources ayant un ecart constant, ici = 1
    Lambda = 1.
    
    # Je demarre avec mes sources melangees/observees
    y=x

    
    # compteur d'affichage
    indice = 1
    
    for I in range(nb_iter):
        DJ = compute_gradient(B,y,x,Lambda)
        # mise a jour de la matrice de separation
        B = B - mu * DJ
        
        # mise a jour d'une estimation des sources separees (approximation dessources avant melange)
        y=B @ x
        
        for i in range(n):
            y[i]=y[i]-np.mean(y[i])
            
        
        
        # Affichage toutes les 100 iterations
        if(I == indice * 100):
            indice = indice + 1
            print('je reconstruis les images separees......')
            yy=[]
            for k in range(n):
                yy.append((y[k]-min(y[k]))/(max(y[k])-min(y[k]))*255)
            image_separees = recons_images_test_nb(yy,nb_lign,nb_col);

            for k in range(n):
                plt.figure(n+k+2)
                plt.clf()
                plt.imshow((np.array(image_separees[k])))
                plt.title('Separation '+str(k))
            
            plt.show()
            plt.pause(5)
            