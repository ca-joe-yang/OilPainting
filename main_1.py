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

stroke_mask = []
for y in range(-R, R):
    for x in range(-R, R):
        if y*y + x*x < R*R:
            stroke_mask.append( (y,x) )

for y in range(height):
    print(y)
    for x in range(width):
        local_histogram = np.zeros(256)
        local_channel_count = np.zeros((channel, 256))
        for dy,dx in stroke_mask:
            yy = y+dy
            xx = x+dx
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

