import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_depth_image_opencv(depth_image_path):
    depth_image = cv2.imread(depth_image_path,cv2.IMREAD_ANYDEPTH)
    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imshow('depth_image', depth_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_depth_image_matplotlib(depth_image_path):
    depth_image = np.load(depth_image_path)
    plt.imshow(depth_image, cmap='viridis')
    plt.colorbar(label='Distance(mm)')
    plt.title("depth image")
    plt.show()

if __name__ == '__main__':
    depth_image_path = ''
    show_depth_image_matplotlib(depth_image_path)
