from flask import Flask, request, jsonify
import requests
import uuid
import threading

app = Flask(__name__)

def process_image_async(image_data, user_image_id):
    try:
        # Character Service로 이미지 데이터와 user_image_id 전송
        requests.post('http://character-service:8081/api/character/process', 
                      json={'image': image_data, 'user_image_id': user_image_id})
    except requests.RequestException as e:
        print(f"Error processing image: {str(e)}")

@app.route('/api/character/create', methods=['POST'])
def create_character():
    image_data = request.json.get('image')
    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400

    # 고유한 user_image_id 생성
    user_image_id = str(uuid.uuid4())

    # 비동기적으로 이미지 처리 요청
    threading.Thread(target=process_image_async, args=(image_data, user_image_id)).start()

    # Frontend에 즉시 user_image_id 반환
    return jsonify({'user_image_id': user_image_id})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)