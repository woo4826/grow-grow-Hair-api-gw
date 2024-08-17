from flask import Flask, request, jsonify
from typing import Dict, Any

app = Flask(__name__)

# 실제 게임 로직은 다른 모듈에서 구현될 예정입니다.
# 여기서는 간단한 더미 함수로 대체합니다.
def dummy_game_logic(action: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "success", "action": action, "data": data}

@app.route('/api/game/start', methods=['POST'])
def start_game():
    data = request.json
    result = dummy_game_logic("start", data)
    return jsonify(result)

@app.route('/api/game/plant', methods=['POST'])
def plant_seed():
    data = request.json
    result = dummy_game_logic("plant", data)
    return jsonify(result)

@app.route('/api/game/water', methods=['POST'])
def water_plant():
    data = request.json
    result = dummy_game_logic("water", data)
    return jsonify(result)

@app.route('/api/game/fertilize', methods=['POST'])
def fertilize_plant():
    data = request.json
    result = dummy_game_logic("fertilize", data)
    return jsonify(result)

@app.route('/api/game/trim', methods=['POST'])
def trim_plant():
    data = request.json
    result = dummy_game_logic("trim", data)
    return jsonify(result)

@app.route('/api/game/finish', methods=['POST'])
def finish_game():
    data = request.json
    result = dummy_game_logic("finish", data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)