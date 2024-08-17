from flask import Flask, request, jsonify
from image_processor import create_bald_character  # 이 함수는 별도로 구현해야 합니다

app = Flask(__name__)

@app.route('/api/character/process', methods=['POST'])
def process_character():
    data = request.json
    image_data = data.get('image')
    user_image_id = data.get('user_image_id')

    if not image_data or not user_image_id:
        return jsonify({'error': 'Missing image data or user_image_id'}), 400

    try:
        # 대머리 캐릭터 생성 로직
        bald_image = create_bald_character(image_data)

        # 여기서 bald_image와 user_image_id를 저장하는 로직이 필요합니다
        # 예: save_character_image(user_image_id, bald_image)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)