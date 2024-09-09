import cv2
import numpy as np
import os

image_path = 'C:/Users/USER/Desktop/OCS-auto-input-deployed/web_service/error_images/50-75%/KakaoTalk_20230530_153032957_15.jpg'
filename, ext = os.path.splitext(os.path.basename(image_path))
src_img = cv2.imread(image_path)
src_img = cv2.resize(src_img, (1000,1000))
point_list = []

COLOR = (255, 0, 255)
THICKNESS = 3
drawing = False

def mouse_handler(event, x, y, flags, param):
    global drawing
    dst_img = src_img.copy()
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        point_list.append((x,y))
        
    if drawing:
        prev_point = None
        for point in point_list:
            cv2.circle(dst_img, point, 15, COLOR, cv2.FILLED)
            if prev_point:
                cv2.line(dst_img, prev_point, point, COLOR, THICKNESS, cv2.LINE_AA)
            prev_point = point
            
        next_point = (x,y)
        if len(point_list) == 4:
            show_result()
            next_point = point_list[0]
            
        cv2.line(dst_img, prev_point, next_point, COLOR, THICKNESS, cv2.LINE_AA)
        
    cv2.imshow('img', dst_img)
    
def show_result():
        src_np = np.array(point_list, dtype=np.float32)
        
        width = max(np.linalg.norm(src_np[0] - src_np[1]),
                    np.linalg.norm(src_np[2] - src_np[3]))
        height = max(np.linalg.norm(src_np[0] - src_np[3]),
                    np.linalg.norm(src_np[1] - src_np[2]))
        
        dst_np = np.array([
            [0,0],
            [width, 0],
            [width, height],
            [0, height],
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src=src_np, dst = dst_np)
        result = cv2.warpPerspective(src_img, M=M, dsize=(int(width), int(height)))
        cv2.imshow('result', result)
        
        result_filename = 'point_image' + ext 
        cv2.imwrite(result_filename, result)
        
cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_handler)
cv2.imshow('img', src_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
