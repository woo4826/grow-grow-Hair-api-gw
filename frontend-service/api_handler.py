from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/api/frontend/capture', methods=['POST'])
def capture_image():
    base64_image = request.json.get('image')
    if not base64_image:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # API Gateway로 이미지 전송
        response = requests.post('http://api-gateway:8080/api/character/create', 
                                 json={'image': base64_image})
        response.raise_for_status()
        data = response.json()
        
        return jsonify({'user_image_id': data['user_image_id']})
    except requests.RequestException as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)