import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

## Constants

IMG_DIR='img/'

LISTE_NOMS=['image1.png', 'image2.png', 'image3.png']

NB_ITER = 20

PAS_AFFICHAGE = NB_ITER//5 # Nombre d'itérations séparant deux affichages

MU = 0.01

LAMBDA = 1

for i in range(len(LISTE_NOMS)):
    LISTE_NOMS[i] = IMG_DIR + LISTE_NOMS[i]


## Programme principal
"""
print("Chargement des images...")

mixed_img_matrix_array = load_img_from_name(LISTE_NOMS)

if (gray_in_list(mixed_img_matrix_array)) :
    print("Conversion des images en gris")
    for k in range(len(mixed_img_matrix_array)) :
        mixed_img_matrix_array[k] = color_to_gray(mixed_img_matrix_array[k])

#print("Conversion des images en vecteurs...")
#
#mixed_img_vect_array, nb_row, nb_col = matrix_to_vect_array(mixed_img_matrix_array)
#
#print("Recomposition des sources à partir des observées...")
#
#y = separate_mixed(mixed_img_vect_array, NB_ITER)

y = mosaique2(mixed_img_matrix_array, 4)

print("Recomposition terminee !")

# Affichage final
#
#clean_img_array = []
#
#for k in range(len(y)):
#    clean_img_array.append((y[k] - min(y[k])) / (max(y[k]) - min(y[k])) * 255)
#
#recomposed_img = vect_to_matrix_array(clean_img_array, nb_row, nb_col)

show_img(y, 1, "Recomposition ")

"""

## Tests
"""
input = load_img_from_name(LISTE_NOMS)

if (gray_in_list(input)) :
    print("Conversion des images en gris")
    for k in range(len(input)) :
        input[k] = color_to_gray(input[k])

#a = create_sub_matrix_array(input, 8, 'mosaic')
#b = revert_sub_matrix_array(a)
#print(b == input)

a, nb_row, nb_col = matrix_to_vect_array(input)
b = separate_mixed(a, NB_ITER)
c = vect_to_matrix_array(b, nb_row, nb_col)

show_img(c, 1, "Input")
"""

input = load_img_from_name(LISTE_NOMS)

mix_img(input[1], input[2], 0.35, IMG_DIR + "marching_on_rails.png")