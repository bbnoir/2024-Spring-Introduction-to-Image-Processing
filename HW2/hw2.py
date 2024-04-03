import cv2 as cv
import numpy as np
import argparse

def compute_histogram(img):
    hist = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1
    return hist

def compute_transform(hist, img):
    trans = np.zeros(256)
    for i in range(1, 256):
        trans[i] = trans[i-1] + hist[i]
    trans = (255.0 * trans / (img.shape[0] * img.shape[1])).round().astype(np.uint8)
    return trans

def inverse_transform(trans):
    inv = np.zeros(256)
    for i in range(256):
        inv[trans[i]] = i
    # make monotonically increasing
    for i in range(1, 256):
        if inv[i] < inv[i-1]:
            inv[i] = inv[i-1]
    return inv

def histogram_equalization(img):
    hist = compute_histogram(img)
    trans = compute_transform(hist, img)
    img2 = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img2[i][j] = trans[img[i][j]]
    return img2

def histogram_specification(img, ref):
    hist = compute_histogram(img)
    trans = compute_transform(hist, img)
    hist_ref = compute_histogram(ref)
    trans_ref = compute_transform(hist_ref, ref)
    trans_ref_inv = inverse_transform(trans_ref)
    img2 = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img2[i][j] = trans_ref_inv[trans[img[i][j]]]
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
