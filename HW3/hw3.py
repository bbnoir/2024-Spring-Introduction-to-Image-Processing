import cv2 as cv
import numpy as np
import argparse
import os

def spatial_filtering(input_image, kernel_type):
    # read input image
    image = cv.imread(input_image, cv.IMREAD_GRAYSCALE)

    # kernel
    identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    if kernel_type == 0:
        kernel = identity - np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif kernel_type == 1:
        kernel = identity - np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    # convolution, padding with edge
    pad = kernel.shape[0] // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), 'edge')
    filtered_image = np.zeros(image.shape, dtype=np.int32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_image[i, j] = np.sum(padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image

def frequency_filtering(input_image):
    # as text book, normalize image to [0, 1]

    # read input image
    image = cv.imread(input_image, cv.IMREAD_GRAYSCALE)

    # normalize
    image_norm = image / 255.0

    # fourier transform
    f = np.fft.fft2(image_norm)
    fshift = np.fft.fftshift(f)

    # Laplacian filter
    P, Q = fshift.shape
    H = np.zeros(fshift.shape)
    for u in range(P):
        for v in range(Q):
            H[u, v] = -4 * np.pi**2 * ((u - P//2)**2 + (v - Q//2)**2)

    # apply filter
    filtered_fshift = fshift * H
    filtered_f = np.fft.ifftshift(filtered_fshift)
    filtered_image = np.real(np.fft.ifft2(filtered_f))

    # scale down filtered image
    scaled_filtered_image = filtered_image / np.max(np.abs(filtered_image))

    # result image
    result_image = image_norm - scaled_filtered_image
    result_image = np.clip(result_image, 0, 1)
    result_image = (result_image * 255).astype(np.uint8)

    return result_image

def main():
    parser = argparse.ArgumentParser(description='HW3 Laplacian Filtering')
    parser.add_argument('-i', '--input', type=str, help='Input image file')
    parser.add_argument('-a', '--all', action='store_true', help='Run spatial filtering & frequency filtering')
    parser.add_argument('-s', '--spatial', action='store_true', help='Run spatial filtering')
    parser.add_argument('-f', '--frequency', action='store_true', help='Run frequency filtering')
    args = parser.parse_args()

    # create result directory if not exist
    if not os.path.exists('results'):
        os.makedirs('results')

    if args.all or args.spatial:
        spatial_filtered_image0 = spatial_filtering(args.input, 0) # type 0, kernel size = 3
        cv.imwrite('results/spatial_filtered_0.png', spatial_filtered_image0)
        spatial_filtered_image1 = spatial_filtering(args.input, 1) # type 1, kernel size = 3
        cv.imwrite('results/spatial_filtered_1.png', spatial_filtered_image1)
    
    if args.all or args.frequency:
        frequency_filtered_image = frequency_filtering(args.input)
        cv.imwrite('results/frequency_filtered.png', frequency_filtered_image)

if __name__ == "__main__":
    main()
