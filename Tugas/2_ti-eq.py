import cv2
import numpy as np

img = cv2.imread('canon.png', 0)
baris, kolom = img.shape
total_pixel = baris * kolom

# 1. Transformasi Intensitas (Negatif)
img_negatif = np.zeros((baris, kolom), dtype=np.uint8)
for i in range(baris):
    for j in range(kolom):
        img_negatif[i, j] = 255 - img[i, j]

# 2. Ekualisasi Histogram   
# a. Hitung intensitas (Histogram)
hist = np.zeros(256, dtype=int)
for i in range(baris):
    for j in range(kolom):
        nilai_pixel = img[i, j]
        hist[nilai_pixel] += 1
            
# b. Hitung CDF (Cumulative Distribution Function)
cdf = np.zeros(256, dtype=int)
cdf[0] = hist[0]
for i in range(1, 256):
    cdf[i] = cdf[i-1] + hist[i]
        
# c. Normalisasi CDF untuk mendapatkan nilai pixel mapping baru
cdf_min = cdf[cdf > 0].min()
mapping = np.zeros(256, dtype=np.uint8)
for i in range(256):
    mapping[i] = round(((cdf[i] - cdf_min) / (total_pixel - cdf_min)) * 255) # Rumus ekualisasi histogram
        
# d. Apply Mapping
img_eq = np.zeros((baris, kolom), dtype=np.uint8)
for i in range(baris):
    for j in range(kolom):
        img_eq[i, j] = mapping[img[i, j]]

# Tampilkan hasil
cv2.imshow('Grayscale', img)
cv2.imshow('Transformasi Intensitas (Negatif)', img_negatif)
cv2.imshow('Ekualisasi Histogram', img_eq)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
