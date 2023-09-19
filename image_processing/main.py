import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np

i = 0
j = 0

image = cv2.imread("img1.jpg")
image1 = image
cv2.waitKey(0)
cv2.destroyAllWindows()
brightness_blue = [0] * 256
brightness_green = [0] * 256
brightness_red = [0] * 256
(wight, height, kanals) = image.shape
print(wight, height)
for i in range(0, wight):
    for j in range(0, height):
        brightness_blue[image[i][j][0]] += 1
        brightness_green[image[i][j][1]] += 1
        brightness_red[image[i][j][2]] += 1
        j += 1
    i += 1

plt.subplot(3, 1, 1)
plt.bar(range(256), brightness_blue)
plt.xlabel('Яркость')
plt.ylabel('Количество цветов')
plt.title('Яркость голубого')
plt.subplot(3, 1, 2)
plt.bar(range(256), brightness_green)
plt.xlabel('Яркость')
plt.ylabel('Количество цветов')
plt.title('Яркость зеленого')
plt.subplot(3, 1, 3)
plt.bar(range(256), brightness_red)
plt.xlabel('Яркость')
plt.ylabel('Количество цветов')
plt.title('Яркость красного')
plt.show()

f_min = int(input("Введите мимнимальное значение старого диапазона: "))
f_max = int(input("Введите максимальное значение старого диапазона: "))
g_min = int(input("Введите минимальное значение нового диапазона: "))
g_max = int(input("Введите миксимальное значение нового диапазона: "))

f = 0
d = 0
k = 0
h = 0

A = (g_max - g_min) / (f_max - f_min)
B = (f_max * g_min - f_min * g_max) / (f_max - f_min)
image2 = numpy.zeros_like(image)

for f in range(0, wight):
    for d in range(0, height):
        image2[f][d][0] = A * image[f][d][0] + B
        image2[f][d][1] = A * image[f][d][1] + B
        image2[f][d][2] = A * image[f][d][2] + B
        d += 1
    f += 1

image3 = numpy.zeros_like(image)
for k in range(0, wight):
    for h in range(0, height):
        image3[k][h][0] = 0.3 * image[k][h][2] + 0.59 * image[k][h][1] + 0.11 * image[k][h][0]
        image3[k][h][1] = 0.3 * image[k][h][2] + 0.59 * image[k][h][1] + 0.11 * image[k][h][0]
        image3[k][h][2] = 0.3 * image[k][h][2] + 0.59 * image[k][h][1] + 0.11 * image[k][h][0]
        h += 1
    k += 1



def filtr2D(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape
    pad_height, pad_width = k_height // 2, k_width // 2
    padded_image = [[0] * (width + pad_width * 2) for _ in range(height + pad_height * 2)]
    for i in range(height):
        for j in range(width):
            padded_image[i + pad_height][j + pad_width] = image[i][j]
    new_image = [[0] * width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            for ki in range(k_height):
                for kj in range(k_width):
                    new_image[i][j] += kernel[ki][kj] * padded_image[i + ki][j + kj]
    return new_image


image4 = cv2.imread('img1.jpg', cv2.IMREAD_GRAYSCALE)

roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)

edge_x = filtr2D(image4, roberts_x)
edge_y = filtr2D(image4, roberts_y)
edges = np.hypot(edge_x, edge_y)

edges = (edges / edges.max()) * 255

cv2.imwrite("edges.jpg", edges)
cv2.destroyAllWindows()

cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.imshow("Preparirovanie", image2)
cv2.waitKey(0)
cv2.imshow("Gray style", image3)
cv2.waitKey(0)
image5 = cv2.imread("edges.jpg")
cv2.imshow("Roberts Operator", image5)
cv2.waitKey(0)
cv2.destroyAllWindows()
