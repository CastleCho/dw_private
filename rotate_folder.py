import os
import cv2
import numpy as np
import time

# 폴더 경로 설정
input_folder_path = 'PHOTO'
output_folder_path = 'after_rotate'

# 폴더 내의 모든 JPG 파일 경로 가져오기
image_paths = [os.path.join(input_folder_path, file) for file in os.listdir(input_folder_path) if file.endswith('.jpg')]

start_time = time.time()

# 각 이미지에 대해 회전 적용
for image_path in image_paths:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges in the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # Find the horizontal lines among the detected lines
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 10:
            horizontal_lines.append((x1, y1, x2, y2))

    # Calculate the mean angle of the horizontal lines for rotation
    mean_angle = np.mean([np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for x1, y1, x2, y2 in horizontal_lines])

    # Get the height and width of the image
    (h, w) = image.shape[:2]

    # Rotate the image
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, mean_angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
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

    # 회전 적용된 이미지 저장
    output_file_name = os.path.basename(image_path)  # 입력 파일명 사용
    output_file_path = os.path.join(output_folder_path, output_file_name)
    cv2.imwrite(output_file_path, final_rotated_image)

end_time = time.time()
execution_time = end_time - start_time
print("Total execution time:", execution_time, "seconds")
