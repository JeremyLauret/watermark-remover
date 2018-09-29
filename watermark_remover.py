import matplotlib.pyplot as plt
import numpy as np
import lib.img_utils as img_utils
import lib.mosaic as mosaic
import lib.separate_mixed as sep
import lib.vect_matrix as vm
import lib.finition_utils as finition
import importlib
# importlib.reload(img_utils)
# importlib.reload(mosaic)
# importlib.reload(sep)
# importlib.reload(vm)
# importlib.reload(finition)


plt.rcParams['image.cmap'] = 'gray'

## Constants

IMG_DIR='img/'

# LISTE_NOMS=['playmobil_on_rails_35.jpg', 'playmobil_on_rails_65.jpg']
LISTE_NOMS=['watermarked_rails.png', 'watermarked_warscene.png', 'warscene.jpg']
# LISTE_NOMS=['watermarked-lena.png', 'watermarked-barbara.png', 'barbara.png']
# LISTE_NOMS=['watermarked_lena.png', 'watermarked_barbara.png', 'barbara.png']

NB_ITER = 100

MU = 0.01

LAMBDA = 1

USE_FINITION = True

for i in range(len(LISTE_NOMS)):
    LISTE_NOMS[i] = IMG_DIR + LISTE_NOMS[i]

print("Chargement des images...")
input_list_m = img_utils.load_img_from_name(LISTE_NOMS) # Supprime l'opacité s'il y en a

# img_utils.mix_img(input_list_m[0], input_list_m[1], 0.35, 'img/playmobil_on_rails_35.jpg')
#
# img_utils.mix_img(input_list_m[0], input_list_m[1], 0.65, 'img/playmobil_on_rails_65.jpg')

if (img_utils.gray_in_list(input_list_m)) :
    print("Conversion des images en gris")
    for k in range(len(input_list_m)) :
        input_list_m[k] = img_utils.color_to_gray(input_list_m[k])

print("Conversion des images en vecteurs...")
input_list_v, nb_row_list, nb_col_list = vm.matrix_to_vect_list(input_list_m)

print("Recomposition des sources à partir des observées...")
output_list_v, dtype_list = sep.separate_mixed(input_list_v, NB_ITER, LAMBDA, MU)

print("Recomposition terminee !")

output_list_v_pre_norm = img_utils.rev_norm(output_list_v) # Inverse la normalisation des valeurs

for i in range(len(output_list_v_pre_norm)):
    if (np.max(output_list_v_pre_norm[i]) > 1):
        output_list_v_pre_norm[i] = np.uint8(output_list_v_pre_norm[i])
        dtype_list[i] = np.dtype(np.uint8)

output_list_m = vm.vect_to_matrix_list(output_list_v_pre_norm, nb_row_list, nb_col_list)

for i in range(len(output_list_m)):
    output_list_m[i] = dtype_list[i].type(output_list_m[i]) # Rétablit le type de données d'origine

img_utils.show_img(output_list_m, 1, "Recomposed ")

if (USE_FINITION):
    # L'image d'intérêt doit être placée en première position dans la liste des noms,
    # l'autre image avec watermark en seconde, et celle sans watermark en troisième.
    finition_list_m = input_list_m.copy()

    # Conversion des matrices de type np.uint8 en type np.float32 afin que le calcul de la pente ait un sens
    for i in range(len(finition_list_m)):
        if (finition_list_m[i].dtype.type == np.uint8):
            finition_list_m[i] = np.float32(finition_list_m[i]/255)

    print('Isolement de la watermark')
    watermark_pixels_list = finition.watermark_pixels(output_list_m, 8)
    print(len(watermark_pixels_list))

    if (len(finition_list_m[0].shape) <= 2):
        for pixel in watermark_pixels_list:
            # On suppose que la transformation appliquée passe par le point de coordonnées (1,1)
            no_division_by_zero = 1 - finition_list_m[1][pixel[0], pixel[1]] if finition_list_m[1][pixel[0], pixel[
                1]] != 1 else np.float32(254 / 255)
            slope = (1 - finition_list_m[2][pixel[0], pixel[1]]) / no_division_by_zero
            finition_list_m[0][pixel[0], pixel[1]] = slope * (finition_list_m[0][pixel[0], pixel[1]] - 1) + 1

    else:
        for pixel in watermark_pixels_list:
            for i in range(finition_list_m[0].shape[2]):
                no_division_by_zero = 1 - finition_list_m[1][pixel[0], pixel[1], i] if finition_list_m[1][pixel[0], pixel[
                    1], i] != 1 else np.float32(254 / 255)
                slope = (1 - finition_list_m[2][pixel[0], pixel[1], i]) / no_division_by_zero
                finition_list_m[0][pixel[0], pixel[1], i] = slope * (finition_list_m[0][pixel[0], pixel[1], i] - 1) + 1
                if (finition_list_m[0][pixel[0], pixel[1], i] < 0):
                    finition_list_m[0][pixel[0], pixel[1], i] = 0
                elif (finition_list_m[0][pixel[0], pixel[1], i] > 1):
                    finition_list_m[0][pixel[0], pixel[1], i] = 1

    img_utils.show_img([finition_list_m[0]], 2, "Finition ")