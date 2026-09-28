import cv2
import numpy as np

img = cv2.imread('canon.png', 0)
baris, kolom = img.shape
    
# Blank image as output placeholder
img_hasil = np.zeros((baris, kolom), dtype=np.uint8)
    
# Kernel Spasial 7x7 (Mean filter)
kernel = np.ones((7, 7)) / 49
                       
# Konvolusi (Filter spasial)
for i in range(3, baris - 3):
    for j in range(3, kolom - 3):
        area = img[i-3 : i+4, j-3 : j+4]
        
        hasil_perhitungan = np.sum(area * kernel)
        
        if hasil_perhitungan < 0:
                hasil_perhitungan = 0
        elif hasil_perhitungan > 255:
                hasil_perhitungan = 255
                
        img_hasil[i, j] = int(hasil_perhitungan)

cv2.imshow('Grayscale', img)
cv2.imshow('Hasil Filter Spasial 7x7 (Manual Blur)', img_hasil)  
cv2.waitKey(0)
cv2.destroyAllWindows()