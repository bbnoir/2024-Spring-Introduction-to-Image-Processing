import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    # plot Equalization
    # equal_in_file = cv.imread('Q1.jpg', cv.IMREAD_GRAYSCALE)
    # equal_out_file = cv.imread('Q1_output.jpg', cv.IMREAD_GRAYSCALE)

    # plot 2 histograms
    # hist_in = cv.calcHist([equal_in_file], [0], None, [256], [0, 256])
    # hist_out = cv.calcHist([equal_out_file], [0], None, [256], [0, 256])
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(hist_in)
    # plt.title('Input Image Histogram')
    # plt.subplot(1, 2, 2)
    # plt.plot(hist_out)
    # plt.title('Output Image Histogram')
    # plt.show()

    # plot 2 CDFs
    # cdf_in = np.cumsum(hist_in)
    # cdf_out = np.cumsum(hist_out)
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(cdf_in)
    # plt.title('Input Image CDF')
    # plt.subplot(1, 2, 2)
    # plt.plot(cdf_out)
    # plt.title('Output Image CDF')
    # plt.show()

    # plot Specified Histogram Equalization
    specified_in_file = cv.imread('Q2_source.jpg', cv.IMREAD_GRAYSCALE)
    specified_ref_file = cv.imread('Q2_reference.jpg', cv.IMREAD_GRAYSCALE)
    specified_out_file = cv.imread('Q2_output.jpg', cv.IMREAD_GRAYSCALE)

    

    hist_in = cv.calcHist([specified_in_file], [0], None, [256], [0, 256])
    hist_ref = cv.calcHist([specified_ref_file], [0], None, [256], [0, 256])
    hist_out = cv.calcHist([specified_out_file], [0], None, [256], [0, 256])
    plt.figure()
    plt.bar(np.arange(256), hist_in.flatten(), color='b', alpha=0.5)
    plt.bar(np.arange(256), hist_ref.flatten(), color='r', alpha=0.5)
    plt.bar(np.arange(256), hist_out.flatten(), color='g', alpha=0.5)
    plt.title('Image Histograms')
    plt.legend(['Input', 'Reference', 'Output'])
    plt.show()
    
    cdf_in = np.cumsum(hist_in)
    cdf_ref = np.cumsum(hist_ref)
    cdf_out = np.cumsum(hist_out)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(cdf_in)
    plt.title('Input Image CDF')
    plt.subplot(1, 3, 2)
    plt.plot(cdf_ref)
    plt.title('Reference Image CDF')
    plt.subplot(1, 3, 3)
    plt.plot(cdf_out)
    plt.title('Output Image CDF')
    plt.show()
    
    plt.figure()
    plt.plot(cdf_in, label='Input Image CDF')
    plt.plot(cdf_ref, label='Reference Image CDF')
    plt.plot(cdf_out, label='Output Image CDF')
    plt.title('Image CDFs')
    plt.legend()
    plt.show()
    
    # show 3 images
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(specified_in_file, cmap='gray')
    plt.title('Input Image')
    plt.subplot(1, 3, 2)
    plt.imshow(specified_ref_file, cmap='gray')
    plt.title('Reference Image')
    plt.subplot(1, 3, 3)
    plt.imshow(specified_out_file, cmap='gray')
    plt.title('Output Image')
    plt.show()
    
    
    
if __name__ == "__main__":
    main()
