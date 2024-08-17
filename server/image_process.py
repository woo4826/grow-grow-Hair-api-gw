import cv2
import numpy as np
import mediapipe as mp
import base64
import io
from PIL import Image
import re

class ImprovedBaldMaker:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def clean_base64(self, base64_string):
        base64_string = re.sub(r'^data:image/.+;base64,', '', base64_string)
        return base64_string.strip().replace(' ', '').replace('\n', '')

    def process_image(self, image_base64):
        image_base64 = self.clean_base64(image_base64)
        
        image_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)

        if not results.detections:
            raise ValueError("No face detected in the image")

        face = results.detections[0]
        bboxC = face.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

        forehead_y = max(0, y - int(h * 0.5))
        skin_color = image[y + h//2, x + w//2]

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hair = np.array([0, 0, 0])
        upper_hair = np.array([30, 255, 150])
        hair_mask = cv2.inRange(hsv_image, lower_hair, upper_hair)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[0:forehead_y, max(0, x-w//2):min(iw, x+w+w//2)] = 255

        combined_mask = cv2.bitwise_or(mask, hair_mask)

        kernel = np.ones((20,20), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        combined_mask = cv2.GaussianBlur(combined_mask, (21, 21), 11)

        bald_image = image.copy()
        bald_image[combined_mask > 128] = skin_color

        _, buffer = cv2.imencode('.png', bald_image)
        bald_image_base64 = base64.b64encode(buffer).decode()

        return bald_image_base64

def make_bald_advanced(image_base64):
    try:
        bald_maker = ImprovedBaldMaker()
        return bald_maker.process_image(image_base64)
    except Exception as e:
        raise ValueError(f"Error in make_bald_advanced: {str(e)}")

def merge_images(bald_image_base64, game_image_base64):
    try:
        bald_maker = ImprovedBaldMaker()
        bald_image_base64 = bald_maker.clean_base64(bald_image_base64)
        game_image_base64 = bald_maker.clean_base64(game_image_base64)

        bald_image = Image.open(io.BytesIO(base64.b64decode(bald_image_base64))).convert('RGBA')
        game_image = Image.open(io.BytesIO(base64.b64decode(game_image_base64))).convert('RGBA')

        bald_array = np.array(bald_image)
        game_array = np.array(game_image)

        # Ensure both arrays have the same shape
        if bald_array.shape != game_array.shape:
            game_array = cv2.resize(game_array, (bald_array.shape[1], bald_array.shape[0]))

        # Ensure both arrays have 4 channels (RGBA)
        if bald_array.shape[2] == 3:
            bald_array = np.dstack((bald_array, np.full(bald_array.shape[:2], 255, dtype=np.uint8)))
        if game_array.shape[2] == 3:
            game_array = np.dstack((game_array, np.full(game_array.shape[:2], 255, dtype=np.uint8)))

        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(bald_array[:,:,:3], cv2.COLOR_RGB2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        hair_mask = 255 - skin_mask

        kernel = np.ones((5,5), np.uint8)
        hair_mask = cv2.dilate(hair_mask, kernel, iterations=2)
        hair_mask = cv2.GaussianBlur(hair_mask, (21, 21), 11)

        hair_mask = hair_mask[:,:,np.newaxis] / 255.0
        hair_mask = np.repeat(hair_mask, 4, axis=2)  # Repeat for RGBA

        result_array = (1 - hair_mask) * bald_array + hair_mask * game_array
        result_array = result_array.astype(np.uint8)

        result_image = Image.fromarray(result_array, 'RGBA')

        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    except Exception as e:
        raise ValueError(f"Error in merge_images: {str(e)}")

# 추가적인 유틸리티 함수들 (필요한 경우)
def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()