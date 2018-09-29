import numpy as np
import matplotlib.pyplot as plt


def no_background_coordinates(matrice, tolerance=20):
    (nb_row, nb_col) = matrice.shape
    copy_matrice = matrice.copy()

    # Passage de l'intervalle [0,1] à [0,255] si nécessaire
    if (copy_matrice.dtype.type != np.uint8):
        copy_matrice = np.uint8(255 * copy_matrice)

    # on recherche la nuance du fond avec une tolérance d'intensité donnée
    nb_shades = 256//tolerance if 256//tolerance == 256/tolerance else 256//tolerance + 1
    shades_list = [0] * nb_shades
    
    for i in range(nb_row):
        for j in range(nb_col):
            shades_list[copy_matrice[i, j] // tolerance] += 1
            
    background_shade = 0
    max = 0

    for shade in range(len(shades_list)):
        if shades_list[shade] > max:
            background_shade = shade
            max = shades_list[shade]

    # construction de la liste des coordonnées des points n'appartenant pas au fond
    coord_list = []

    for i in range(nb_row):
        for j in range(nb_col):
            if (int(copy_matrice[i, j]) // tolerance != background_shade):
                coord_list.append([i, j])
                
    return coord_list


def watermark_pixels(output_list_m, tolerance):
    no_background_coord_list = []
    color = False

    if (len(output_list_m[0].shape) > 2):
        nb_col = output_list_m[0].shape[2]
        color = True

    if (color):
        for i in range(len(output_list_m)):
            for j in range(nb_col):
                no_background_coord_list.append(no_background_coordinates(output_list_m[i][:,:,j], tolerance))

    else:
        for i in range(len(output_list_m)):
            no_background_coord_list.append(no_background_coordinates(output_list_m[i], tolerance))

    # on choisit comme pixels de texte la liste la plus courte (hypothèse que la watermark a le fond le plus vaste)
    watermark_pixels_list = no_background_coord_list[0]
    min = len(no_background_coord_list[0])

    for i in range(len(no_background_coord_list)):
        if len(no_background_coord_list[i]) < min:
            min = len(no_background_coord_list[i])
            watermark_pixels_list = no_background_coord_list[i]

    return watermark_pixels_list