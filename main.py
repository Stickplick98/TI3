"""
from tkinter import Tk     #pip install tk
from tkinter.filedialog import askopenfilename
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("C:\\TraitementImages3eme\\Images\\lenaAEgaliser.jpg",cv.IMREAD_COLOR)

hist_b = cv.calcHist([img], [0], None, [256], [0, 256])
hist_g = cv.calcHist([img], [1], None, [256], [0, 256])
hist_r = cv.calcHist([img], [2], None, [256], [0, 256])

# Create a matplotlib figure with subplots
fig, axs = plt.subplots(2, 4, figsize=(15, 10))

# Display the original image
axs[0, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axs[0, 0].set_title('Image de base')

#Display the first Histogramm
axs[1, 0].plot(hist_b, color='blue', label='Bleu')
axs[1, 0].plot(hist_g, color='green', label='Vert')
axs[1, 0].plot(hist_r, color='red', label='Rouge')
axs[1, 0].set_title('Histogramme couleur')


equalized_channels = [cv.equalizeHist(channel) for channel in cv.split(img)]
equalized_image = cv.merge(equalized_channels)

#cv.imshow("image modifié",equalized_image)

hist_b = cv.calcHist([equalized_image], [0], None, [256], [0, 256])
hist_g = cv.calcHist([equalized_image], [1], None, [256], [0, 256])
hist_r = cv.calcHist([equalized_image], [2], None, [256], [0, 256])

# Display the original image
axs[0, 1].imshow(cv.cvtColor(equalized_image, cv.COLOR_BGR2RGB))
axs[0, 1].set_title('Image couleur égalisée')

#Display the first Histogramm
axs[1, 1].plot(hist_b, color='blue', label='Bleu')
axs[1, 1].plot(hist_g, color='green', label='Vert')
axs[1, 1].plot(hist_r, color='red', label='Rouge')
axs[1, 1].set_title('Histogramme couleur égalisé')

#//////////////////////////////////////////////////////

# Convertir l'image en espace de couleur HSV
hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cv.imshow("Image de base", hsv_image)

# Display the original image
axs[0, 2].imshow(hsv_image)
axs[0, 2].set_title('Image luminance')
h,s,v = cv.split(hsv_image)
hist_equalized_luminance = cv.calcHist([v], [0], None, [256], [0, 256])

axs[1, 2].plot(hist_equalized_luminance, color='red', label='Luminance')
axs[1, 2].set_title("Histogramme de la composante de luminance")


# Égaliser l'histogramme de la composante de luminance (V)
v2 = cv.equalizeHist(v)

histV = cv.calcHist([v2], [0], None, [256], [0, 256])

plt.plot(histV, color='red', label='Rouge egalisé')
plt.title("test")
plt.xlabel("Niveau d'intensité")
plt.ylabel("Fréquence")
plt.legend()
plt.show()

hist_equalized_luminanceV2 = cv.merge((h,s,v))
# Reconvertir l'image en espace de couleur BGR
hist_equalized_luminanceV2 = cv.cvtColor(hsv_image, cv.COLOR_HSV2RGB)

#cv.imshow("Image modifiée", equalized_image)

# Afficher l'histogramme de l'image égalisée en niveaux de gris
hist_equalized_luminancev2 = cv.calcHist(hsv_image, [2], None, [256], [0, 256])

axs[0, 3].imshow(cv.cvtColor(equalized_image, cv.COLOR_BGR2RGB))
axs[0, 3].set_title('Image luminance égalisée')
axs[1, 3].plot(hist_equalized_luminancev2, color='red', label='Luminance')
axs[1, 3].set_title("Histogramme de la composante de luminance")

plt.plot(hist_equalized_luminance, color='red', label='Luminance égalisée')
plt.title("Histogramme de la composante de luminance égalisée")

# Display the combined plot
plt.tight_layout()
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
"""

"""
from tkinter import Tk     #pip install tk
from tkinter.filedialog import askopenfilename
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("C:\\TraitementImages3eme\\Images\\lenaAEgaliser.jpg",cv.IMREAD_COLOR)

hist_b = cv.calcHist([img], [0], None, [256], [0, 256])
hist_g = cv.calcHist([img], [1], None, [256], [0, 256])
hist_r = cv.calcHist([img], [2], None, [256], [0, 256])

cv.imshow("image de base",img)
plt.plot(hist_b, color='blue', label='Bleu')
plt.plot(hist_g, color='green', label='Vert')
plt.plot(hist_r, color='red', label='Rouge')
plt.title("Histogramme de l'image couleur")
plt.xlabel("Niveau d'intensité")
plt.ylabel("Fréquence")
plt.legend()
plt.show()

equalized_channels = [cv.equalizeHist(channel) for channel in cv.split(img)]
equalized_image = cv.merge(equalized_channels)

cv.imshow("image modifié",equalized_image)

hist_b = cv.calcHist([equalized_image], [0], None, [256], [0, 256])
hist_g = cv.calcHist([equalized_image], [1], None, [256], [0, 256])
hist_r = cv.calcHist([equalized_image], [2], None, [256], [0, 256])

plt.plot(hist_b, color='blue', label='Bleu egalisé')
plt.plot(hist_g, color='green', label='Vert egalisé')
plt.plot(hist_r, color='red', label='Rouge egalisé')
plt.title("Histogramme de l'image couleur égalisé")
plt.xlabel("Niveau d'intensité")
plt.ylabel("Fréquence")
plt.legend()
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()

# Convertir l'image en espace de couleur HSV
hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cv.imshow("Image de base", hsv_image)

hist_luminance = cv.calcHist([hsv_image[:,:,2]], [0], None, [256], [0, 256])
plt.plot(hist_luminance, color='red', label='Luminance')
plt.title("Histogramme de la composante de luminance")
plt.xlabel("Niveau d'intensité")
plt.ylabel("Fréquence")
plt.legend()
plt.show()

# Égaliser l'histogramme de la composante de luminance (V)
hsv_image[:,:,2] = cv.equalizeHist(hsv_image[:,:,2])

# Reconvertir l'image en espace de couleur BGR
equalized_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

cv.imshow("Image modifiée", equalized_image)

# Afficher l'histogramme de l'image égalisée en niveaux de gris
hist_equalized_luminance = cv.calcHist([hsv_image[:,:,2]], [0], None, [256], [0, 256])
plt.plot(hist_equalized_luminance, color='red', label='Luminance égalisée')
plt.title("Histogramme de la composante de luminance égalisée")
plt.xlabel("Niveau d'intensité")
plt.ylabel("Fréquence")
plt.legend()
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
"""


import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("C:\\TraitementImages3eme\\Images\\lenaAEgaliser.jpg",cv.IMREAD_COLOR)

hist_b = cv.calcHist([img], [0], None, [256], [0, 256])
hist_g = cv.calcHist([img], [1], None, [256], [0, 256])
hist_r = cv.calcHist([img], [2], None, [256], [0, 256])

equalized_channels = [cv.equalizeHist(channel) for channel in cv.split(img)]
equalized_image = cv.merge(equalized_channels)

hist_be = cv.calcHist([equalized_image], [0], None, [256], [0, 256])
hist_ge = cv.calcHist([equalized_image], [1], None, [256], [0, 256])
hist_re = cv.calcHist([equalized_image], [2], None, [256], [0, 256])

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

hist_h = cv.calcHist([img_hsv], [0], None, [256], [0, 256])
hist_s = cv.calcHist([img_hsv], [1], None, [256], [0, 256])
hist_v = cv.calcHist([img_hsv], [2], None, [256], [0, 256])

h,s,v = cv.split(img_hsv)
hist_ve = cv.equalizeHist(v)

img_hsv_equalized = cv.merge((h,s,hist_ve))

hist_h_equalized = cv.calcHist([img_hsv_equalized], [0], None, [256], [0, 256])
hist_s_equalized = cv.calcHist([img_hsv_equalized], [1], None, [256], [0, 256])
hist_v_equalized = cv.calcHist([img_hsv_equalized], [2], None, [256], [0, 256])

fig, axs = plt.subplots(2, 4, figsize=(10, 5))

axs[0, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axs[0, 0].set_title('Image de base')

axs[1, 0].plot(hist_b, color='blue', label='Bleu')
axs[1, 0].plot(hist_g, color='green', label='Vert')
axs[1, 0].plot(hist_r, color='red', label='Rouge')
axs[1, 0].set_title('Histogramme couleur')

axs[0, 1].imshow(cv.cvtColor(equalized_image, cv.COLOR_BGR2RGB))
axs[0, 1].set_title('Image couleur égalisée')

axs[1, 1].plot(hist_be, color='blue', label='Bleu')
axs[1, 1].plot(hist_ge, color='green', label='Vert')
axs[1, 1].plot(hist_re, color='red', label='Rouge')
axs[1, 1].set_title('Histogramme couleur égalisé')

axs[0, 2].imshow(cv.cvtColor(img_hsv,cv.COLOR_HSV2RGB))
axs[0, 2].set_title('Image de base 2')

axs[1, 2].plot(hist_v, color='red', label='Luminance')
axs[1, 2].set_title("Histogramme de la composante de luminance")

axs[0, 3].imshow(cv.cvtColor(img_hsv_equalized, cv.COLOR_HSV2RGB))
axs[0, 3].set_title('Image luminance égalisée')

axs[1, 3].plot(hist_v_equalized, color='red', label='Luminance')
axs[1, 3].set_title("Histogramme de la composante de luminance")



plt.tight_layout()
plt.show()
