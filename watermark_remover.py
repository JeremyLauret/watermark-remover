import matplotlib.pyplot as plt
import numpy as np
import lib.img_utils as img_utils
import lib.mosaic as mosaic
import lib.separate_mixed as sep
import lib.vect_matrix as vm
import importlib
importlib.reload(img_utils)
importlib.reload(mosaic)
importlib.reload(sep)
importlib.reload(vm)


plt.rcParams['image.cmap'] = 'gray'

## Constants

IMG_DIR='img/'

# LISTE_NOMS=['watermarked_lena.png', 'watermarked_barbara.png', 'barbara.png']
LISTE_NOMS=['war_on_rails_35.jpg', 'war_on_rails_65.jpg']

NB_ITER = 1000

PAS_AFFICHAGE = NB_ITER//5 # Nombre d'itérations séparant deux affichages

MU = 0.01

LAMBDA = 1

for i in range(len(LISTE_NOMS)):
    LISTE_NOMS[i] = IMG_DIR + LISTE_NOMS[i]

print("Chargement des images...")
input_list_m = img_utils.load_img_from_name(LISTE_NOMS) # Supprime l'opacité s'il y en a

# img_utils.mix_img(input_list_m[0], input_list_m[1], 0.35, 'img/war_on_rails_35.jpg')

# img_utils.mix_img(input_list_m[0], input_list_m[1], 0.65, 'img/war_on_rails_65.jpg')

if (img_utils.gray_in_list(input_list_m)) :
    print("Conversion des images en gris")
    for k in range(len(input_list_m)) :
        input_list_m[k] = img_utils.color_to_gray(input_list_m[k])

print("Conversion des images en vecteurs...")
input_list_v, nb_row_list, nb_col_list = vm.matrix_to_vect_list(input_list_m)

print("Recomposition des sources à partir des observées...")
output_list_v, dtype = sep.separate_mixed(input_list_v, NB_ITER, LAMBDA, MU)

print("Recomposition terminee !")

output_list_v_pre_norm = img_utils.rev_norm(output_list_v) # Inverse la normalisation des valeurs

output_list_m = vm.vect_to_matrix_list(output_list_v_pre_norm, nb_row_list, nb_col_list)

for i in range(len(output_list_m)):
    output_list_m = dtype.type(output_list_m) # Rétablit le type de données d'origine

img_utils.show_img(output_list_m, 1, "Mixed ")