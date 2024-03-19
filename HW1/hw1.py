import cv2 as cv
import numpy as np
import argparse

def bicubic_calc(p, x):
    return (-p[0] + 3 * p[1] - 3 * p[2] + p[3]) / 2.0 * (x ** 3) \
        + (2 * p[0] - 5 * p[1] + 4 * p[2] - p[3]) / 2.0 * (x ** 2) \
        + (-p[0] + p[2]) / 2.0 * x + p[1]

def rotate(img, angle, method):
    h, w, c = img.shape
    result = np.zeros((h, w, c), np.uint8)
    pivot_x, pivot_y = w / 2, h / 2
    angle = np.deg2rad(-angle)
    rotate_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                              [np.sin(angle), np.cos(angle), 0],
                              [0, 0, 1]])
    translate_matrix = np.array([[1, 0, -pivot_x],
                             [0, 1, -pivot_y],
                             [0, 0, 1]])
    inv_translate_matrix = np.array([[1, 0, pivot_x],
                                [0, 1, pivot_y],
                                [0, 0, 1]])
    transform_matrix = np.dot(inv_translate_matrix, np.dot(rotate_matrix, translate_matrix))

    if method == 'nearest':
        for i in range(h):
            for j in range(w):
                x, y, _ = np.dot(transform_matrix, [j, i, 1])
                x, y = int(round(x)), int(round(y))
                if x >= 0 and x < w and y >= 0 and y < h:
                    result[i, j] = img[y, x]
    elif method == 'bilinear':
        for i in range(h):
            for j in range(w):
                x, y, _ = np.dot(transform_matrix, [j, i, 1])
                if x >= 0 and x < w-1 and y >= 0 and y < h-1:
                    dx, dy = x - int(x), y - int(y)
                    top = (1 - dx) * img[int(y), int(x)] + dx * img[int(y), int(x + 1)]
                    bot = (1 - dx) * img[int(y + 1), int(x)] + dx * img[int(y + 1), int(x + 1)]
                    result[i, j] = (1 - dy) * top + dy * bot
    elif method == 'bicubic':
        img = img.astype(np.int32)
        for i in range(h):
            for j in range(w):
                x, y, _ = np.dot(transform_matrix, [j, i, 1])
                if x >= 0 and x < w and y >= 0 and y < h:
                    x1, y1 = int(x), int(y)
                    x0, y0 = max(x1 - 1, 0), max(y1 - 1, 0)
                    x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
                    x3, y3 = min(x1 + 2, w - 1), min(y1 + 2, h - 1)
                    dx, dy = x - x1, y - y1
                    q1 = bicubic_calc([img[y0, x0], img[y0, x1], img[y0, x2], img[y0, x3]], dx)
                    q2 = bicubic_calc([img[y1, x0], img[y1, x1], img[y1, x2], img[y1, x3]], dx)
                    q3 = bicubic_calc([img[y2, x0], img[y2, x1], img[y2, x2], img[y2, x3]], dx)
                    q4 = bicubic_calc([img[y3, x0], img[y3, x1], img[y3, x2], img[y3, x3]], dx)
                    res = bicubic_calc([q1, q2, q3, q4], dy)
                    np.clip(res, 0, 255, out=res)
                    result[i, j] = res.astype(np.uint8)
    return result

def enlarge(img, scale, method):
    h, w, c = img.shape
    result = np.zeros((h * scale, w * scale, c), np.uint8)
    if method == 'nearest':
        for i in range(h * scale):
            for j in range(w * scale):
                result[i, j] = img[i // scale, j // scale]
    elif method == 'bilinear':
        for i in range(h * scale):
            for j in range(w * scale):
                x = j / scale
                y = i / scale
                x1, y1 = int(x), int(y)
                x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
                dx, dy = x - x1, y - y1
                top = (1 - dx) * img[y1, x1] + dx * img[y1, x2]
                bot = (1 - dx) * img[y2, x1] + dx * img[y2, x2]
                result[i, j] = (1 - dy) * top + dy * bot
    elif method == 'bicubic':
        img = img.astype(np.int32) # convert to int32 to avoid overflow
        for i in range(h * scale):
            for j in range(w * scale):
                x = j / scale
                y = i / scale
                x1, y1 = int(x), int(y)
                x0, y0 = max(x1 - 1, 0), max(y1 - 1, 0)
                x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
                x3, y3 = min(x1 + 2, w - 1), min(y1 + 2, h - 1)
                dx, dy = x - x1, y - y1
                q1 = bicubic_calc([img[y0, x0], img[y0, x1], img[y0, x2], img[y0, x3]], dx)
                q2 = bicubic_calc([img[y1, x0], img[y1, x1], img[y1, x2], img[y1, x3]], dx)
                q3 = bicubic_calc([img[y2, x0], img[y2, x1], img[y2, x2], img[y2, x3]], dx)
                q4 = bicubic_calc([img[y3, x0], img[y3, x1], img[y3, x2], img[y3, x3]], dx)
                res = bicubic_calc([q1, q2, q3, q4], dy)
                np.clip(res, 0, 255, out=res)
                result[i, j] = res.astype(np.uint8)
    else:
        raise ValueError('Invalid method')
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--all', action='store_true', help='run all functions')
    parser.add_argument('-i', '--image', default='building.jpg', help='path to the image')
    parser.add_argument('-t', '--type', default='enlarge', help='rotate or enlarge')
    parser.add_argument('-m', '--method', default='bicubic', help='nearest or bilinear or bicubic')
    args = parser.parse_args()
    img = cv.imread(args.image)

    if args.all:

        result = enlarge(img, 2, 'nearest')
        filename = args.image.split('.')[0] + '_enlarge_nearest.jpg'
        cv.imwrite(filename, result)

        result = enlarge(img, 2, 'bilinear')
        filename = args.image.split('.')[0] + '_enlarge_bilinear.jpg'
        cv.imwrite(filename, result)

        result = enlarge(img, 2, 'bicubic')
        filename = args.image.split('.')[0] + '_enlarge_bicubic.jpg'
        cv.imwrite(filename, result)

        result = rotate(img, 30, 'nearest')
        filename = args.image.split('.')[0] + '_rotate_nearest.jpg'
        cv.imwrite(filename, result)

        result = rotate(img, 30, 'bilinear')
        filename = args.image.split('.')[0] + '_rotate_bilinear.jpg'
        cv.imwrite(filename, result)

        result = rotate(img, 30, 'bicubic')
        filename = args.image.split('.')[0] + '_rotate_bicubic.jpg'
        cv.imwrite(filename, result)

    elif args.type == 'rotate':

        result = rotate(img, 30, args.method)
        filename = args.image.split('.')[0] + '_' + args.type + '_' + args.method + '.jpg'
        cv.imwrite(filename, result)

    elif args.type == 'enlarge':

        result = enlarge(img, 2, args.method)
        filename = args.image.split('.')[0] + '_' + args.type + '_' + args.method + '.jpg'
        cv.imwrite(filename, result)
        
    else:
        raise ValueError('Invalid type')

if __name__ == "__main__":
    main()
