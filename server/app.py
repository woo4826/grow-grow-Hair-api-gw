# import binascii
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uuid
# import base64
# from PIL import Image
# import io
# import cv2
# import numpy as np
# import uvicorn
# import logging

# app = FastAPI()

# # 임시 데이터 저장소 (실제 구현에서는 데이터베이스를 사용해야 합니다)
# user_images = {}
# game_states = {}

# class UserImage(BaseModel):
#     image: str  # base64 encoded image

# class GameState(BaseModel):
#     user_id: str
#     game_image: str  # base64 encoded game result image

# def create_bald_image(image_data):
#     # base64 문자열 정리
#     image_data = image_data.strip()  # 앞뒤 공백 제거
#     image_data = image_data.replace(' ', '')  # 모든 공백 제거
#     image_data = image_data.replace('\n', '')  # 줄바꿈 제거
    
#     # 패딩 추가 (필요한 경우)
#     missing_padding = len(image_data) % 4
#     if missing_padding:
#         image_data += '=' * (4 - missing_padding)
    
#     logging.debug(f"Base64 string length: {len(image_data)}")
#     logging.debug(f"First 100 characters: {image_data[:100]}")

#     try:
#         # base64 디코딩
#         image_bytes = base64.b64decode(image_data)
#     except binascii.Error as e:
#         raise ValueError(f"Invalid base64 string: {str(e)}")

#     # 바이트 데이터를 numpy 배열로 변환
#     nparr = np.frombuffer(image_bytes, np.uint8)
    
#     # numpy 배열을 이미지로 디코딩
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     if img is None:
#         raise ValueError("Failed to decode image")

#     # 이미지 처리 로직
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x+w, int(y-h/3)), (80, 80, 80), -1)

#     # 처리된 이미지를 다시 base64로 인코딩
#     _, buffer = cv2.imencode('.jpg', img)
#     return base64.b64encode(buffer).decode('utf-8')

# @app.post("/start_game")
# async def start_game(user_image: UserImage):
#     try:
#         bald_image = create_bald_image(user_image.image)
#         user_id = str(uuid.uuid4())
#         user_images[user_id] = bald_image
#         return {"user_id": user_id, "bald_image": bald_image}
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/finish_game")
# async def finish_game(game_state: GameState):
#     if game_state.user_id not in user_images:
#         raise HTTPException(status_code=404, detail="User not found")
    
#     bald_image_base64 = user_images[game_state.user_id]
#     game_image_base64 = game_state.game_image

#     try:
#         # base64 디코딩 및 이미지로 변환
#         bald_image = Image.open(io.BytesIO(base64.b64decode(bald_image_base64)))
#         game_image = Image.open(io.BytesIO(base64.b64decode(game_image_base64)))

#         # PIL Image를 numpy 배열로 변환
#         bald_array = np.array(bald_image)
#         game_array = np.array(game_image)

#         # 이미지 크기 맞추기
#         game_array = cv2.resize(game_array, (bald_array.shape[1], bald_array.shape[0]))

#         # 머리 부분 찾기 (흰색 영역)
#         mask = cv2.inRange(bald_array, (75, 75, 75), (80, 80, 80))

#         # 게임 이미지를 마스크를 사용하여 합성
#         result_array = np.where(mask[:,:,None], game_array, bald_array)

#         # numpy 배열을 다시 PIL Image로 변환
#         result_image = Image.fromarray(result_array)

#         # 결과 이미지를 base64로 인코딩
#         buffered = io.BytesIO()
#         result_image.save(buffered, format="PNG")
#         final_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

#         return {"final_image": final_image_base64}
    
#     except Exception as e:
#         logging.error(f"Error in finish_game: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"An error occurred while processing the images: {str(e)}")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import binascii
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import base64
from PIL import Image
import io
import cv2
import numpy as np
import uvicorn
import logging
import dlib

app = FastAPI()

# 임시 데이터 저장소 (실제 구현에서는 데이터베이스를 사용해야 합니다)
user_images = {}
game_states = {}

class UserImage(BaseModel):
    image: str  # base64 encoded image

class GameState(BaseModel):
    user_id: str
    game_image: str  # base64 encoded game result image

def make_bald_advanced(base64_image):
    try:
        # Base64 이미지 디코딩
        image_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_data, np.uint8)
        open_cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if open_cv_image is None:
            raise ValueError("Failed to decode image")

        # dlib의 얼굴 감지기와 랜드마크 예측기 로드
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        
        # 얼굴 감지
        faces = detector(open_cv_image)
        
        for face in faces:
            # 얼굴 랜드마크 검출
            landmarks = predictor(open_cv_image, face)
            
            # 이마의 상단 경계 계산 (눈썹 위 20% 정도)
            forehead_top = landmarks.part(27).y - int(0.2 * (face.bottom() - face.top()))
            forehead_top = max(forehead_top, 0)  # 이미지 경계 확인
            
            # 이마의 좌우 경계 계산
            left = landmarks.part(0).x
            right = landmarks.part(16).x
            
            # 피부색 추출 (코 주변에서 샘플링)
            nose_bridge_color = open_cv_image[landmarks.part(27).y, landmarks.part(27).x]
            
            # 그라데이션 효과를 위한 마스크 생성
            mask = np.zeros(open_cv_image.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (left, forehead_top), (right, face.top()), 255, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 11)
            
            # 대머리 효과 적용
            balding_effect = open_cv_image.copy()
            balding_effect[:] = nose_bridge_color
            
            # 그라데이션 적용
            open_cv_image = cv2.seamlessClone(
                balding_effect, open_cv_image, mask, 
                (int((left + right) / 2), int((forehead_top + face.top()) / 2)),
                cv2.NORMAL_CLONE
            )
        
        # 결과 이미지를 Base64로 인코딩
        _, buffer = cv2.imencode('.png', open_cv_image)
        return base64.b64encode(buffer).decode()
    except Exception as e:
        logging.error(f"Error in make_bald_advanced: {str(e)}")
        raise

@app.post("/start_game")
async def start_game(user_image: UserImage):
    try:
        bald_image = make_bald_advanced(user_image.image)
        user_id = str(uuid.uuid4())
        user_images[user_id] = bald_image
        return {"user_id": user_id, "bald_image": bald_image}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/finish_game")
async def finish_game(game_state: GameState):
    if game_state.user_id not in user_images:
        raise HTTPException(status_code=404, detail="User not found")
    
    bald_image_base64 = user_images[game_state.user_id]
    game_image_base64 = game_state.game_image

    try:
        # base64 디코딩 및 이미지로 변환
        bald_image = Image.open(io.BytesIO(base64.b64decode(bald_image_base64)))
        game_image = Image.open(io.BytesIO(base64.b64decode(game_image_base64)))

        # PIL Image를 numpy 배열로 변환
        bald_array = np.array(bald_image)
        game_array = np.array(game_image)

        # 이미지 크기 맞추기
        game_array = cv2.resize(game_array, (bald_array.shape[1], bald_array.shape[0]))

        # 머리 부분 찾기 (피부색 영역)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        hsv = cv2.cvtColor(bald_array, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 게임 이미지를 마스크를 사용하여 합성
        result_array = np.where(mask[:,:,None], game_array, bald_array)

        # numpy 배열을 다시 PIL Image로 변환
        result_image = Image.fromarray(result_array)

        # 결과 이미지를 base64로 인코딩
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        final_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {"final_image": final_image_base64}
    
    except Exception as e:
        logging.error(f"Error in finish_game: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the images: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(app, host="0.0.0.0", port=8000)