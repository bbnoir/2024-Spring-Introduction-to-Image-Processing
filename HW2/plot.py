import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse

def compute_histogram(img):
    hist = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1
    return hist

def compute_cdf(hist, img):
    cdf = np.zeros(256)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    cdf = (255.0 * cdf / (img.shape[0] * img.shape[1])).round().astype(np.uint8)
    return cdf

def main():
    # plot Equalization
    equal_in_file = cv.imread('Q1.jpg', cv.IMREAD_GRAYSCALE)
    equal_out_file = cv.imread('Q1_output.png', cv.IMREAD_GRAYSCALE)

    # plot 2 images without axis
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(equal_in_file, cmap='gray')
    plt.title('Source Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(equal_out_file, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')
    plt.savefig('plots/equal_images.png')

    # plot 2 histograms
    hist_in = compute_histogram(equal_in_file)
    hist_out = compute_histogram(equal_out_file)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(256), hist_in, color='blue')
    plt.title('Source Histogram')
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(256), hist_out, color='orange')
    plt.title('Equalized Histogram')
    plt.savefig('plots/equal_histograms.png')

    # plot 2 CDFs
    cdf_in = compute_cdf(hist_in, equal_in_file)
    cdf_out = compute_cdf(hist_out, equal_out_file)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cdf_in, color='blue')
    plt.title('Source CDF')
    plt.subplot(1, 2, 2)
    plt.plot(cdf_out, color='orange')
    plt.title('Equalized CDF')
    plt.savefig('plots/equal_cdfs.png')

    # plot Specified Histogram Equalization
    specified_in_file = cv.imread('Q2_source.jpg', cv.IMREAD_GRAYSCALE)
    specified_ref_file = cv.imread('Q2_reference.jpg', cv.IMREAD_GRAYSCALE)
    specified_out_file = cv.imread('Q2_output.png', cv.IMREAD_GRAYSCALE)

    hist_in = compute_histogram(specified_in_file)
    hist_ref = compute_histogram(specified_ref_file)
    hist_out = compute_histogram(specified_out_file)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.bar(np.arange(256), hist_in, color='blue')
    plt.title('Source Histogram')
    plt.subplot(1, 3, 2)
    plt.bar(np.arange(256), hist_ref, color='green')
    plt.title('Reference Histogram')
    plt.subplot(1, 3, 3)
    plt.bar(np.arange(256), hist_out, color='orange')
    plt.title('Specified Histogram')
    plt.savefig('plots/spec_histograms.png')
    
    
    cdf_in = compute_cdf(hist_in, specified_in_file)
    cdf_ref = compute_cdf(hist_ref, specified_ref_file)
    cdf_out = compute_cdf(hist_out, specified_out_file)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(cdf_in, color='blue')
    plt.title('Source CDF')
    plt.subplot(1, 3, 2)
    plt.plot(cdf_ref, color='green')
    plt.title('Reference CDF')
    plt.subplot(1, 3, 3)
    plt.plot(cdf_out, color='orange')
    plt.title('Specified CDF')
    plt.savefig('plots/spec_cdfs.png')

    # show 3 images
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(specified_in_file, cmap='gray')
    plt.title('Source Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(specified_ref_file, cmap='gray')
    plt.title('Reference Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(specified_out_file, cmap='gray')
    plt.title('Specified Image')
    plt.axis('off')
    plt.savefig('plots/spec_images.png')
    
if __name__ == "__main__":
    main()
