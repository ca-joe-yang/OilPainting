#! python3

import cv2
import sys
import numpy as np

image_filename = sys.argv[1]
color_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)

height = color_image.shape[0]
width  = color_image.shape[1]
channel = color_image.shape[2]

R = 4

gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
oil_image = np.zeros((height, width, channel))

M_0 = np.average(gray_image)
N = 32
L = 8
threshold = 0.2

def moment(x,y,r,angle):
    s = 0
    xx = x + int(r * np.cos(angle))
    yy = y + int(r * np.sin(angle))
    if yy < 0 or yy >= height or xx <= 0  or xx >= width:
        s = gray_image[yy, xx]
    return s

local_CLD = np.zeros((height, width, N))
for i in range(N):
    angle = i * 2 * np.pi / N
    for y in range(height):
        print(i, y)
        for x in range(width):
            l = 0
            s = gray_image[y, x]
            while(True):
                l += 1
                xx = x + int(l * np.cos(angle))
                yy = y + int(l * np.sin(angle))
                if not (yy < 0 or yy >= height or xx <= 0  or xx >= width):
                    s += gray_image[yy, xx]
                if (s/l - M_0) / M_0 <= threshold:
                    local_CLD[y,x,i] = l
                    break

brushstroke_size = np.zeros((height, width))
for y in range(height):
    for x in range(width):
        brushstroke_size[y,x] = N * L / np.sum(local_CLD[y,x])

for y in range(height):
    print(y)
    for x in range(width):
        R = int(brushstroke_size[y,x])
        local_histogram = np.zeros(256)
        local_channel_count = np.zeros((channel, 256))
        for dy in range(-R, R):
            for dx in range(-R, R):
                yy = y+dy
                xx = x+dx
                if dy*dy + dx*dx > R*R:
                    continue
                if yy < 0  or yy >= height or xx <= 0  or xx >= width:
                    continue
                intensity = gray_image[yy, xx]
                local_histogram[intensity] += 1
                for c in range(channel):
                    local_channel_count[c, intensity] += color_image[yy, xx, c]

        max_intensity = np.argmax(local_histogram)
        max_intensity_count = local_histogram[max_intensity]
        for c in range(channel):
            oil_image[y,x,c] = local_channel_count[c, max_intensity] / max_intensity_count

oil_image = oil_image.astype('int')
cv2.imwrite("result.jpg", oil_image)
