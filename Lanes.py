import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.fastNlMeansDenoising(gray, None, 25, 7, 21)
    canny_img = cv2.Canny(blur, 100, 200)
    dilate = cv2.dilate(canny_img, kernel=None, iterations=0)
    return dilate

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    # Define a trapezoidal region of interest that covers only the left half of the image
    left_top = (int(width * 0.45), int(height * 0.6))
    left_bottom = (int(width * 0.1), int(height))
    right_top = (int(width * 0.55), int(height * 0.6))
    right_bottom = (int(width * 0.9), int(height))
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Load the image
image = cv2.imread('WhiteLine.jpg')

# Process the image
lane_image = image.copy()
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
line_image = display_lines(lane_image, lines)

# Display the results
fig, axs = plt.subplots(1, 2, figsize=(15, 15))
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[1].imshow(canny_image)
axs[1].set_title('Lane Lines Detected')
plt.show()
