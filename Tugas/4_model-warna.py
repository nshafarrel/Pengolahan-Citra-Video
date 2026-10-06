import cv2
import numpy as np


# 1. BGR - CMYK
def bgr_to_cmyk(img_bgr):
    b = img_bgr[:, :, 0].astype(np.float64) / 255.0
    g = img_bgr[:, :, 1].astype(np.float64) / 255.0
    r = img_bgr[:, :, 2].astype(np.float64) / 255.0
    
    k = 1.0 - np.maximum(np.maximum(r, g), b)
    
    c = np.where(k == 1.0, 0, (1.0 - r - k) / (1.0 - k))
    m = np.where(k == 1.0, 0, (1.0 - g - k) / (1.0 - k))
    y = np.where(k == 1.0, 0, (1.0 - b - k) / (1.0 - k))
    
    return np.stack([c, m, y, k], axis=-1)

def cmyk_to_bgr(img_cmyk):
    c, m, y, k = cv2.split(img_cmyk)
    
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    
    bgr = np.stack([b, g, r], axis=-1) * 255.0
    return bgr.astype(np.uint8)

# 2. BGR - HSI
def bgr_to_hsi(img_bgr):
    b = img_bgr[:, :, 0].astype(np.float64) / 255.0
    g = img_bgr[:, :, 1].astype(np.float64) / 255.0
    r = img_bgr[:, :, 2].astype(np.float64) / 255.0
    
    pembilang = 0.5 * ((r - g) + (r - b))
    penyebut = np.sqrt((r - g)**2 + (r - b) * (g - b))
    
    rasio = np.where(penyebut > 1e-12, pembilang / penyebut, 0)
    theta = np.degrees(np.arccos(np.clip(rasio, -1.0, 1.0)))
    
    h = np.where(b <= g, theta, 360.0 - theta)
    
    jumlah = r + g + b
    minimum = np.minimum(np.minimum(r, g), b)
    s = np.where(jumlah == 0, 0, 1.0 - (3.0 * minimum / jumlah))
    i = jumlah / 3.0
    
    return np.stack([h, s, i], axis=-1)

def hsi_to_bgr(img_hsi):
    h, s, i = cv2.split(img_hsi)
    
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    
    mask_rg = (h >= 0) & (h < 120)
    h_rg = np.radians(h[mask_rg])
    b[mask_rg] = i[mask_rg] * (1.0 - s[mask_rg])
    r[mask_rg] = i[mask_rg] * (1.0 + (s[mask_rg] * np.cos(h_rg) / np.cos(np.radians(60) - h_rg)))
    g[mask_rg] = 3.0 * i[mask_rg] - (r[mask_rg] + b[mask_rg])
    
    mask_gb = (h >= 120) & (h < 240)
    h_gb = np.radians(h[mask_gb] - 120.0)
    r[mask_gb] = i[mask_gb] * (1.0 - s[mask_gb])
    g[mask_gb] = i[mask_gb] * (1.0 + (s[mask_gb] * np.cos(h_gb) / np.cos(np.radians(60) - h_gb)))
    b[mask_gb] = 3.0 * i[mask_gb] - (r[mask_gb] + g[mask_gb])
    
    mask_br = (h >= 240) & (h <= 360)
    h_br = np.radians(h[mask_br] - 240.0)
    g[mask_br] = i[mask_br] * (1.0 - s[mask_br])
    b[mask_br] = i[mask_br] * (1.0 + (s[mask_br] * np.cos(h_br) / np.cos(np.radians(60) - h_br)))
    r[mask_br] = 3.0 * i[mask_br] - (g[mask_br] + b[mask_br])
    
    bgr = np.stack([b, g, r], axis=-1) * 255.0
    return np.clip(bgr, 0, 255).astype(np.uint8)

# 3. BGR - HSV
def bgr_to_hsv(img_bgr):
    b = img_bgr[:, :, 0].astype(np.float64) / 255.0
    g = img_bgr[:, :, 1].astype(np.float64) / 255.0
    r = img_bgr[:, :, 2].astype(np.float64) / 255.0
    
    m_max = np.maximum(np.maximum(r, g), b)
    m_min = np.minimum(np.minimum(r, g), b)
    delta = m_max - m_min
    
    v = m_max
    s = np.where(m_max == 0, 0, delta / m_max)
    h = np.zeros_like(r)
    
    mask_r = (m_max == r) & (delta != 0)
    mask_g = (m_max == g) & (delta != 0)
    mask_b = (m_max == b) & (delta != 0)
    
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)
    
    return np.stack([h, s, v], axis=-1)

def hsv_to_bgr(img_hsv):
    h, s, v = cv2.split(img_hsv)
    
    c = v * s
    x = c * (1.0 - np.abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
    
    mask0 = (h >= 0) & (h < 60)
    r[mask0], g[mask0], b[mask0] = c[mask0], x[mask0], 0
    
    mask1 = (h >= 60) & (h < 120)
    r[mask1], g[mask1], b[mask1] = x[mask1], c[mask1], 0
    
    mask2 = (h >= 120) & (h < 180)
    r[mask2], g[mask2], b[mask2] = 0, c[mask2], x[mask2]
    
    mask3 = (h >= 180) & (h < 240)
    r[mask3], g[mask3], b[mask3] = 0, x[mask3], c[mask3]
    
    mask4 = (h >= 240) & (h < 300)
    r[mask4], g[mask4], b[mask4] = x[mask4], 0, c[mask4]
    
    mask5 = (h >= 300) & (h < 360)
    r[mask5], g[mask5], b[mask5] = c[mask5], 0, x[mask5]
    
    bgr = np.stack([b + m, g + m, r + m], axis=-1) * 255.0
    return np.clip(bgr, 0, 255).astype(np.uint8)

# main
if __name__ == "__main__":
    img = cv2.imread('canon.png')
    
    if img is not None:
        cv2.imshow("Original BGR", img)
        
        cmyk_img = bgr_to_cmyk(img)
        hsi_img = bgr_to_hsi(img)
        hsv_img = bgr_to_hsv(img)
        
        cv2.imshow("CMYK to BGR Recovered", cmyk_to_bgr(cmyk_img))
        cv2.imshow("HSI to BGR Recovered", hsi_to_bgr(hsi_img))
        cv2.imshow("HSV to BGR Recovered", hsv_to_bgr(hsv_img))
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Pastikan file 'canon.png' berada di dalam direktori yang sama.")