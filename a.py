import numpy as np 
# this is the key library for manipulating arrays. Use the online ressources! http://www.numpy.org/

import matplotlib.pyplot as plt 
# used to read images, display and plot http://matplotlib.org/api/pyplot_api.html

import scipy.ndimage as ndimage
# one of several python libraries for image procession

plt.rcParams['image.cmap'] = 'gray' 

def watermark(Img,W,alpha):
    image = plt.imread(Img)
    if image.shape[2] == 3:
        image = image[:,:,2]
    watermark = plt.imread(W)
    if watermark.shape[2] == 3:
        watermark = image[:,:,2]
    result = alpha * watermark + (1 - alpha) * image
    plt.imshow(result)
    plt.imsave('watermarked='+Img,result)
    return


def compute_gradient(B,y1,y2,x1,x2,lam1):
    m_y1 = np.mean(y1)
    m_y1_2 = np.mean(y1**2)    
    m_y1_3 = np.mean(y1**3)    
    m_y1_4 = np.mean(y1**4)    
    m_y1_5 = np.mean(y1**5)    
    m_y1_6 = np.mean(y1**6)            
    m_y2 = np.mean(y2)
    m_y2_2 = np.mean(y2**2)     
    m_y2_3 = np.mean(y2**3)     
    m_y2_4 = np.mean(y2**4)    
    m_y2_5 = np.mean(y2**5)     
    m_y2_6 = np.mean(y2**6)         

    #K1 = np.array([1, m_y1, m_y1_2, m_y1_3])
    #K2 = np.array([1, m_y2, m_y2_2, m_y2_3])
    
    # M1 = K1.T @ K1
    M1 = np.array([[1, m_y1, m_y1_2, m_y1_3], [m_y1, m_y1_2, m_y1_3, m_y1_4], [m_y1_2, m_y1_3, m_y1_4, m_y1_5], [m_y1_3, m_y1_4, m_y1_5, m_y1_6]])
    # M1 = K2.T @ K2
    M2 = np.array([[1, m_y2, m_y2_2, m_y2_3], [m_y2, m_y2_2, m_y2_3, m_y2_4], [m_y2_2, m_y2_3, m_y2_4, m_y2_5], [m_y2_3, m_y2_4, m_y2_5, m_y2_6]])


    P1 = np.array([[0, 1, 2*m_y1, 3*(m_y1_2)]])
    P2 = np.array([[0, 1, 2*m_y2, 3*(m_y2_2)]])
    

    w1 = np.linalg.inv(M1) @ P1.T
    w2 = np.linalg.inv(M2) @ P2.T
    
   
    Psi_y1 = w1[0]+w1[1]*y1+w1[2]*y1**2+w1[3]*y1**3
    Psi_y2 = w2[0]+w2[1]*y2+w2[2]*y2**2+w2[3]*y2**3

    #Psi_y = [Psi_y1, Psi_y2]
    
    M_Psi11 = np.mean(Psi_y1*x1)
    M_Psi12 = np.mean(Psi_y1*x2)
    M_Psi21 = np.mean(Psi_y2*x1)
    M_Psi22 = np.mean(Psi_y2*x2)
    
  
    y1 = y1 - np.mean(y1)
    y2 = y2 - np.mean(y2)
    
    
    temp1 = 4 * (np.mean(y1**2) - 1) * y1
    temp2 = 4 * (np.mean(y2**2) - 1) * y2

    m_y1_x1 = np.mean(temp1 * x1)
    m_y1_x2 = np.mean(temp1 * x2)
    m_y2_x1 = np.mean(temp2 * x1)
    m_y2_x2 = np.mean(temp2 * x2)
    pen = np.array([[m_y1_x1, m_y1_x2], [m_y2_x1, m_y2_x2]])

    return(np.array([[M_Psi11, M_Psi12], [M_Psi21, M_Psi22]]) @ B.T - np.eye(2) + lam1 * pen @ B.T)


def gene_images_test_nb():

    #***** lecture de l'image par exemple: xxx.jpg
    
    image_source11 = (plt.imread('W.png')) / 255
    
    image_source1 = image_source11[:, :, 2]
    # cas limite double(rgb2gray(plt.imread('ponts.png')))/255
    
    image_source2 = (plt.imread('barbara.png')) / 255
    
    # cas limite double(rgb2gray(plt.imread('chaussees.png')))/255

    
    ##****** dimensions de l'image (nb_lign,nb_col,3)

    [nb_lign, nb_col] = image_source1.shape
      
      
    #*************************** separation des composantes RGB
    R1 = image_source1[:,:]
    R2 = image_source2[:,:]
    #***** deconstruction de l'image a partir des composantes en des vecteurs
    R_L1 = []
    R_L2 = []

    for j in range(nb_lign):
        for i in range(nb_col):
            R_L1.append(R1[j,i])  
            R_L2.append(R2[j,i])

    #**** conversion double precision pour la regle de trois Fs double()
    R_L1 = np.array(R_L1)
    R_L2 = np.array(R_L2)
    #***************** regle de trois pour se ramener dans l'intervalle [0,1]
     
    return [R_L1, R_L2, nb_lign, nb_col, image_source1, image_source2]

    
def recons_images_test_nb(y1_R, y2_R, nb_lign, nb_col):
    image_R1 = np.zeros((nb_lign, nb_col))
    image_R2 = np.zeros((nb_lign, nb_col))
    for j in range(nb_lign):
        image_R1[j,:] = y1_R[ nb_col*j : nb_col*(j+1)]
        image_R2[j,:] = y2_R[ nb_col*j : nb_col*(j+1)]

    #image_reconst1[:,:] = image_R1

    #image_reconst2[:,:] = image_R2
     
    #image1 = uint8(image_reconst1);
    #image2 = uint8(image_reconst2);
    return [image_R1, image_R2]

# Ne fonctionne pas encore 
def correl_coef_composante_nb(im1_R, im2_R):
    N = len(im1_R)

    moy_1R = np.mean(im1_R)
    moy_2R = np.mean(im2_R)

    Mat_cor_R = np.zeros((2,2))
    #Mat_cor_G = np.zeros((2,2))
    #Mat_cor_B = np.zeros((2,2))
    
    ec_1R = np.std(im1_R)
    ec_2R = np.std(im2_R)
    
    ec_R = np.array([[ec_1R, ec_2R]]).T
    
    moy_R = np.array([[moy_1R, moy_2R]]).T
                   
    ima_R = [im1_R[:,:], im2_R[:,:] ] #erreur : faudrait-il utiliser np.concatenate ?

    for i in range(2):
        for j in range(2):
            Mat_cor_R[i,j] = (1/(N*ec_R(i)*ec_R(j))) * sum(  ( ima_R[:,i] - moy_R[i] ) * ( ima_R[:,j] - moy_R[j] ) )
    return Mat_cor_R


# Programme principal
def main():
    #####
    print("Generation puis concatenation des images en vecteur .....")
    
    # s1, s2 vecteurs sources
    [s1, s2, nb_lign, nb_col, image_source1, image_source2] = gene_images_test_nb()
    
    s1 = s1 - np.mean(s1)
    s2 = s2 - np.mean(s2)

    s1 = s1 / np.std(s1)
    s2 = s2 / np.std(s2)
    
    #x1_R = s1
    #x2_R = s2
    
    # melange
    A11 = 0.6
    A12 = 0.4
    A21 = 0.35 # 1 - A11 ?
    A22 = 0.65 # 1 - A12 ? 
    x1 = A11*s1+A12*s2
    x2 = A21*s1+A22*s2
    
    #plt.figure(1)
    #plt.imshow(image_source1)
    #plt.figure(2)
    #plt.imshow(image_source2)
    
    plt.figure(1)
    plt.subplot(221)
    plt.title("Source 1")
    plt.imshow(image_source1)
    plt.subplot(222)
    plt.title("Source 2")
    plt.imshow(image_source2)
    
    #####
    print('Reconstruction des images de melanges......')
    xx1=(x1-np.min(x1)) / (np.max(x1)-np.min(x1)) * 255
    xx2=(x2-np.min(x2)) / (np.max(x2)-np.min(x2)) * 255

    [image_mel1, image_mel2] = recons_images_test_nb(xx1,xx2,nb_lign,nb_col)
    
    plt.subplot(223)
    plt.imshow((image_mel1))
    plt.title('Melange 1') 
    
    plt.subplot(224)
    plt.imshow((image_mel2))
    plt.title('Melange 2') 
    

    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)

    x1 = x1 / np.std(x1)
    x2 = x2 / np.std(x2)

    
    print('l algo tourne.......')

    nb_iter = 2000 
    # Initialisation de la matrice de separation
    B = np.eye(2)
    
    # pas dans la descente du gradient
    mu=0.01
    
    # hyperparametre : parametre de penalisation : je cherche des sources ayant un ecart constant, ici = 1
    Lambda = 1.
    
    # Je demarre avec mes sources melangees/observees
    y1=x1
    y2=x2
    
    # compteur d'affichage
    indice = 1
    
    for i in range(nb_iter):
        DJ = compute_gradient(B,y1,y2,x1,x2,Lambda)
        # mise a jour de la matrice de separation
        B = B - mu * DJ
        
        # mise a jour d'une estimation des sources separees (approximation dessources avant melange)
        y1 = B[0,0] * x1 + B[0,1] * x2
        y2 = B[1,1] * x2 + B[1,0] * x1

        y1 = y1 - np.mean(y1)
        y2 = y2 - np.mean(y2)
        
        
        # Affichage toutes les 50 iterations
        if(i == indice * 50):
            indice = indice + 1
            print('je reconstruis les images separees......')
            yy1=(y1-min(y1))/(max(y1)-min(y1))*255
            yy2=(y2-min(y2))/(max(y2)-min(y2))*255
            [image_sep1,image_sep2] = recons_images_test_nb(yy1,yy2,nb_lign,nb_col)

            plt.figure(5)
            plt.clf()
            plt.imshow((image_sep1))
            plt.title('Separation 1')
            
            plt.figure(6)
            plt.clf()
            plt.imshow((image_sep2))
            plt.title('Separation 2')

            plt.show()
            plt.pause(1)
            
            """
            # Calcul de la correlation entre les sources avant melange
            [Mat_or_cor_source] = correl_coef_composante_nb(s1,s2)
            plt.pause(1)
            # Calcul de la correlation entre les sources melangees
            [Mat_mel_cor] = correl_coef_composante_nb(x1,x2)
            # Calcul de la correlation entre les sources separees
            [Mat_sep_cor] = correl_coef_composante_nb(y1,y2)
            plt.pause(5)
            """
            
def main2():
    #***** lecture de l'image par exemple: xxx.jpg
    
    image_source11 = (plt.imread('W.png')) / 255
    
    image_source1 = image_source11[:, :, 2]

    #image_source1 = 1 - image_source1
    # cas limite double(rgb2gray(plt.imread('ponts.png')))/255
    



    #image_source2 = (plt.imread('watermarked-barbara.png')) / 255
    
    image_source21 = (plt.imread('barbara.png')) / 255

    image_source22 = image_source11[:,:,2]

    image_source2 = 0.7 * image_source21 + 0.3 * image_source22
    

    image_source1 = 0.5 * image_source1 + 0.5 * image_source2
    
    
    
    #image_source1 = 0.3 * image_source3 + 0.7 * image_source1
    
    plt.figure(10)
    plt.imshow(image_source1)
    plt.show()
    
    plt.figure(9)
    image_source31 =  (plt.imread('barbara.png')) / 255
    image_source32 =  image_source11[:,:,2]
    image_source3 = 0.35 * image_source31 + 0.65 * image_source32
    plt.imshow(image_source3)
    plt.show()
    
    
    
    
    print(image_source3) 
    print('  ')
    print(image_source1)
    
    
    # cas limite double(rgb2gray(plt.imread('chaussees.png')))/255

    
    ##****** dimensions de l'image (nb_lign,nb_col,3)

    [nb_lign, nb_col] = image_source1.shape
      
    
      
    #*************************** separation des composantes RGB
    R1 = image_source1[:,:]
    R2 = image_source2[:,:]
    #***** deconstruction de l'image a partir des composantes en des vecteurs
    R_L1 = []
    R_L2 = []

    for j in range(nb_lign):
        for i in range(nb_col):
            R_L1.append(R1[j,i])  
            R_L2.append(R2[j,i])

    #**** conversion double precision pour la regle de trois Fs double()
    R_L1 = np.array(R_L1)
    R_L2 = np.array(R_L2)
    #***************** regle de trois pour se ramener dans l'intervalle [0,1]
    
    x1 = R_L1
    x2 = R_L2
    
    print(x1)
    
    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)

    x1 = x1 / np.std(x1)
    x2 = x2 / np.std(x2)

    
    print('l algo tourne.......')

    nb_iter = 2000 
    # Initialisation de la matrice de separation
    B = np.eye(2)
    
    # pas dans la descente du gradient
    mu=0.01
    
    # hyperparametre : parametre de penalisation : je cherche des sources ayant un ecart constant, ici = 1
    Lambda = 1.
    
    # Je demarre avec mes sources melangees/observees
    y1=x1
    y2=x2
    
    # compteur d'affichage
    indice = 1
    
    for i in range(nb_iter):
        DJ = compute_gradient(B,y1,y2,x1,x2,Lambda)
        # mise a jour de la matrice de separation
        B = B - mu * DJ
        
        # mise a jour d'une estimation des sources separees (approximation dessources avant melange)
        y1 = B[0,0] * x1 + B[0,1] * x2
        y2 = B[1,1] * x2 + B[1,0] * x1

        y1 = y1 - np.mean(y1)
        y2 = y2 - np.mean(y2)
        
        
        # Affichage toutes les 50 iterations
        if(i == indice * 100):
            indice = indice + 1
            print('je reconstruis les images separees......')
            yy1=(y1-min(y1))/(max(y1)-min(y1))*255
            yy2=(y2-min(y2))/(max(y2)-min(y2))*255
            [image_sep1,image_sep2] = recons_images_test_nb(yy1,yy2,nb_lign,nb_col)

            plt.pause(0.1)
            plt.figure(5)
            plt.clf()
            plt.imshow((image_sep1))
            plt.title('Separation 1')
            
            plt.figure(6)
            plt.clf()
            plt.imshow((image_sep2))
            plt.title('Separation 2')

            plt.show()
            
def test():
    image_source11 = (plt.imread('W.png'))
    
    image_source1 = image_source11[:, :, 2]

    #image_source1 = 1 - image_source1
    # cas limite double(rgb2gray(plt.imread('ponts.png')))/255
    



    #image_source2 = (plt.imread('watermarked-barbara.png')) / 255
    
    image_source21 = (plt.imread('barbara.png'))

    image_source22 = image_source11[:,:,2]

    image_source2 = 0.7 * image_source21 + 0.3 * image_source22
    

    image_source1 = 0.5 * image_source1 + 0.5 * image_source2
    
    
    plt.imsave('test.png',image_source1)
    
    I = plt.imread('test.png')

    print(I) 
    print('  ')
    print(image_source1)
    
    
    
    return I == image_source1
    
            
