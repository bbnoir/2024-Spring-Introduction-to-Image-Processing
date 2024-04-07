import cv2 as cv
import numpy as np
import argparse

def compute_histogram(img):
    hist = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1
    return hist

def compute_cdf(hist, img):
    cdf = np.zeros(256)
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    cdf = (255.0 * cdf / (img.shape[0] * img.shape[1])).round().astype(np.uint8)
    return cdf

def histogram_equalization(img):
    hist = compute_histogram(img)
    cdf = compute_cdf(hist, img)
    img2 = np.zeros_like(img)
    img2 = cdf[img]
    return img2

def gen_spec_map(cdf_ref):
    ref_map = np.zeros(256)
    for i in range(256):
        ref_map[i] = np.argmin(np.abs(cdf_ref - i))
    return ref_map

def histogram_specification(img, ref):
    hist_source = compute_histogram(img)
    cdf_source = compute_cdf(hist_source, img)
    hist_ref = compute_histogram(ref)
    cdf_ref = compute_cdf(hist_ref, ref)
    ref_map = gen_spec_map(cdf_ref)
    img2 = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img2[i, j] = ref_map[cdf_source[img[i, j]]]
    return img2

def main():
    parser = argparse.ArgumentParser(description='HW2')
    parser.add_argument('-a', '--all', action='store_true', help='Run all parts')
    parser.add_argument('-e', '--equal', action='store_true', help='Histogram Equalization')
    parser.add_argument('-s', '--spec', action='store_true', help='Histogram Specification')

    args = parser.parse_args()

    equal_in_file = cv.imread('Q1.jpg', cv.IMREAD_GRAYSCALE)
    spec_in_file = cv.imread('Q2_source.jpg', cv.IMREAD_GRAYSCALE)
    spec_ref_file = cv.imread('Q2_reference.jpg', cv.IMREAD_GRAYSCALE)

    if args.all or args.equal:
        equal_out_file = histogram_equalization(equal_in_file)
        cv.imwrite('Q1_output.jpg', equal_out_file)
    
    if args.all or args.spec:
        spec_out_file = histogram_specification(spec_in_file, spec_ref_file)
        cv.imwrite('Q2_output.jpg', spec_out_file)

if __name__ == "__main__":
    main()
