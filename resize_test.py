import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_and_display(image, size=(256, 256)):
    """
    画像を指定されたサイズにリサイズして表示する関数
    """
    # 画像をリサイズ
    resized_image = cv2.resize(image, size)
    
    # 画像を表示
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Resized to {size[0]}x{size[1]}")
    plt.axis('off')
    plt.show()

def resize_with_aspect_ratio_and_display(image, target_size=(256, 256)):
    """
    アスペクト比を維持して画像をリサイズし、パディングを追加して表示する関数
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # アスペクト比を維持してリサイズ
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    # パディングを追加して目標サイズに合わせる
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    padded_image = cv2.copyMakeBorder(resized_image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 画像を表示
    plt.imshow(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Resized with aspect ratio to {target_size[0]}x{target_size[1]}")
    plt.axis('off')
    plt.show()

# 画像を読み込み
img = cv2.imread('MSRC_ObjCategImageDatabase_v2/Images/20_6_s.bmp')

# 256x256にリサイズして表示
resize_and_display(img)

# アスペクト比を維持してリサイズし表示
resize_with_aspect_ratio_and_display(img)