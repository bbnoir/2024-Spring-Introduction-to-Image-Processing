import cv2 as cv
import numpy as np

def main():
    crop_x_start, crop_x_end = 0, 60
    crop_y_start, crop_y_end = 180, 240
    filename = './building_enlarge_nearest.jpg'
    img = cv.imread(filename)
    img_crop = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    cv.imwrite('./building_enlarge_nearest_crop.jpg', img_crop)
    filename = './building_enlarge_bilinear.jpg'
    img = cv.imread(filename)
    img_crop = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    cv.imwrite('./building_enlarge_bilinear_crop.jpg', img_crop)
    filename = './building_enlarge_bicubic.jpg'
    img = cv.imread(filename)
    img_crop = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    cv.imwrite('./building_enlarge_bicubic_crop.jpg', img_crop)
    cv.imshow('crop', img_crop)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
