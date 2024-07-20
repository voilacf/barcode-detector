import matplotlib.pyplot as plt
import numpy as np
import cv2


def detect_and_draw_barcode(img):
    # Converting given image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Applying the Gaussian Blur
    gray = cv2.GaussianBlur(gray, ksize=(9, 9), sigmaX=0)
    # Computing the Scharr-Gradient-Magnitude-Representation
    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # Subtracting the y-gradient from the x-gradient
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)
    plt.imshow(img)

    # Blurring the image
    blurr = cv2.blur(gradient, (9, 9))
    # Thresholding the image
    (_, threshold) = cv2.threshold(blurr, 225, 255, cv2.THRESH_BINARY)
    # Constructing a closing kernel and applying it to the threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    # Performing several iterations of erosion and dilation
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    plt.imshow(img)
    plt.show()

    # Extracting the contours in the image
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sorting found contours and keeping the largest
    lg_cntr = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # Computing the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(lg_cntr)
    box = np.intp(cv2.boxPoints(rect))

    # Drawing a green quadrilateral around the detected barcode
    cv2.drawContours(img, [box], -1, (0, 255, 0), 10)
    plt.imshow(img)
    plt.show()


def main():
    # detecting the barcode of each image file within the project
    # and drawing the quadrilateral around it
    for i in range(5):
        image = cv2.imread("barcode-{}.tif".format(i))
        detect_and_draw_barcode(image)

# execute to run project
if __name__ == "__main__":
    main()
