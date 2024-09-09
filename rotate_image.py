import cv2
import numpy as np
import time

start_time = time.time()
image_path = "test.jpg"
#test
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect edges in the image
edges = cv2.Canny(gray, 50,150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Find the horizontal lines among the detected lines
horizontal_lines = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(y2 - y1) < 10:
        horizontal_lines.append((x1, y1, x2, y2))

# Calculate the mean angle of the horizontal lines for rotation
mean_angle = np.mean([np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for x1, y1, x2, y2 in horizontal_lines])
print("rotate: ", mean_angle)

# Get the height and width of the image
(h, w) = image.shape[:2]

# Rotate the image
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, mean_angle, 1.0)
rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Detect edges in the rotated image
rotated_gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
rotated_edges = cv2.Canny(rotated_gray, 50, 150, apertureSize=3)

# Detect lines in the rotated image
rotated_lines = cv2.HoughLinesP(rotated_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Find the horizontal lines among the detected lines in the rotated image
rotated_horizontal_lines = []
for line in rotated_lines:
    x1, y1, x2, y2 = line[0]
    if abs(y2 - y1) < 10:
        rotated_horizontal_lines.append((x1, y1, x2, y2))

# Calculate the mean angle of the horizontal lines in the rotated image for rotation
rotated_mean_angle = np.mean([np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for x1, y1, x2, y2 in rotated_horizontal_lines])
print("rotate again: ", rotated_mean_angle)

# Rotate the image again
M = cv2.getRotationMatrix2D(center, rotated_mean_angle, 1.0)
final_rotated_image = cv2.warpAffine(rotated_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
cv2.imwrite('corrected_image.jpg', final_rotated_image)

# import cv2
# import numpy as np
# import base64

# def save_image(image, filename):
#     cv2.imwrite(filename, image)

# def verify_image(image_np, form_format):
#     image_bin_res = cv2.imencode('.jpg', image_np)[1].tobytes()
#     image_bin_res = base64.b64encode(image_bin_res).decode('utf-8')
    
#     if form_format == 0:
#         image = image_np.copy()
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#         save_image(edges, 'edges_image.jpg')
        
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines for detected edges
        
#         horizontal_lines = []
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             if abs(y2 - y1) < 10:
#                 horizontal_lines.append((x1, y1, x2, y2))
        
#         mean_angle = np.mean([np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for x1, y1, x2, y2 in horizontal_lines])

#         # Draw a normal green line as a reference (horizontal line)
#         h, w = image.shape[:2]
#         cv2.line(image, (0, h//2), (w, h//2), (0, 255, 0), 2)  # Green line
        
#         save_image(image, 'detected_lines_image.jpg')

#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, mean_angle, 1.0)
#         final_rotated_image = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#         save_image(final_rotated_image, 'final_rotated_image.jpg')
        
#         image_bin_res = cv2.imencode('.jpg', final_rotated_image)[1].tobytes()
#         image_bin_res = base64.b64encode(image_bin_res).decode('utf-8')
#         return image_bin_res

# # 테스트 이미지 로드
# image_np = cv2.imread('form no. 1-9.jpg')

# # 이미지 전처리 및 저장
# verify_image(image_np, 0)
