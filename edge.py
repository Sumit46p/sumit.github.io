import numpy as np
import matplotlib.pyplot as plt
import cv2


# Load and resize image
im_path = "bill.jpg"
img = cv2.imread(im_path)
img = cv2.resize(img, (1500, 800))


# Display original image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.show()

# Convert to grayscale
orig = img.copy()
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)


# Adaptive thresholding for edge enhancement
gray = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2
)

# Morphological operations
kernel = np.ones((5,5), np.uint8)
gray = cv2.dilate(gray, kernel, iterations=2)
gray = cv2.erode(gray, kernel, iterations=1)

# Gaussian blur
blurred = cv2.GaussianBlur(gray, (5,5), 0)


# Canny edge detection
edge = cv2.Canny(blurred, 10, 100)
orig_edge = edge.copy()

# Contour detection
contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found:", len(contours))

# Sort and select target contour
contours = sorted(contours, reverse=True, key=cv2.contourArea)
target = None

for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * p, True)
    if len(approx) == 4 and cv2.contourArea(c) > 50000:
        target = approx
        break

# Handle fallback if no quadrilateral is found
if target is None:
    print("No suitable quadrilateral contour found! Using largest contour as fallback.")
    if contours:
        largest_contour = contours[0]
        p = cv2.arcLength(largest_contour, True)
        epsilon = 0.02 * p
        while True:
            target = cv2.approxPolyDP(largest_contour, epsilon, True)
            if len(target) == 4 or epsilon > 0.5 * p:
                break
            epsilon += 0.01 * p

        if len(target) != 4:
            print("Fallback contour still not a quadrilateral. Manually selecting 4 points.")
            if len(largest_contour) >= 4:
                indices = np.linspace(0, len(largest_contour) - 1, 4, dtype=int)
                target = largest_contour[indices]
            else:
                print("Not enough points in contour to form a quadrilateral. Aborting.")
                target = np.array([[[0,0]], [[0,800]], [[800,800]], [[800,0]]])
    else:
        print("No contours found. Aborting.")
        target = np.array([[[0,0]], [[0,800]], [[800,800]], [[800,0]]])

print("Target shape:", target.shape)

# Reorder contour points
def reorder(h):
    h = h.reshape((4,2))
    print("Original points:\n", h)
    hnew = np.zeros((4,2), dtype=np.float32)

    add = h.sum(axis=1)
    hnew[0] = h[np.argmin(add)]      # Top-left
    hnew[2] = h[np.argmax(add)]      # Bottom-right

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]     # Top-right
    hnew[3] = h[np.argmax(diff)]     # Bottom-left

    return hnew

reordered = reorder(target)
print("Reordered points:\n", reordered)

# Perspective transform
input_representation = reordered
output_map = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])

m = cv2.getPerspectiveTransform(input_representation, output_map)
ans = cv2.warpPerspective(orig, m, (800, 800))

# Display final result
plt.imshow(cv2.cvtColor(ans, cv2.COLOR_BGR2RGB))
plt.title("Warped Perspective Output")
plt.show()
