# app.py

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/app.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os

# --- 새로운 프레임워크의 컴포넌트 임포트 ---
from .main_backtest import run_backtest_from_config
from .config_loader import load_config

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)


@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/run_backtest', methods=['POST'])
def run_backtest_endpoint():
    """웹 UI로부터 백테스트 요청을 받아 실행하고 결과를 반환합니다."""
    
    # 1. 웹 UI에서 전송된 파라미터를 받음
    form_data = request.json
    
    # 2. 기본 config/config.yaml 설정을 불러옴
    try:
        config = load_config()
    except Exception as e:
        return jsonify({"error": f"설정 파일 로딩 실패: {e}"}), 500

    # 3. UI에서 받은 값으로 기본 설정을 덮어씀
    #    (HTML 폼의 name 속성과 config.yaml의 키를 일치시키는 것이 좋음)
    config['backtest_settings']['start_date'] = form_data.get('start_date', config['backtest_settings']['start_date'])
    config['backtest_settings']['end_date'] = form_data.get('end_date', config['backtest_settings']['end_date'])
    config['backtest_settings']['initial_cash'] = int(form_data.get('initial_cash', config['backtest_settings']['initial_cash']))
    
    # 전략 파라미터 덮어쓰기
    config['strategy_params']['max_stocks'] = int(form_data.get('max_stocks', config['strategy_params']['max_stocks']))
    # ... UI에 있는 다른 전략 파라미터들도 동일하게 덮어쓰기 ...

    # 4. 수정된 config를 사용하여 새로운 백테스팅 함수 호출
    result = run_backtest_from_config(config)

    # 5. 결과를 JSON 형태로 프론트엔드에 반환
    if "error" in result:
        return jsonify(result), 400
    
    return jsonify(result)

@app.route('/results/<path:subpath>', methods=['GET'])
def download_file(subpath):
    """
    결과 파일(이미지, CSV)을 다운로드할 수 있도록 서빙합니다.
    예: /results/run_20231027_123456/equity_curve.png
    """
    # 보안을 위해 실제 파일 시스템 경로를 직접 노출하지 않도록 주의
    # 여기서는 간단하게 구현
    base_dir = os.path.abspath("results")
    file_path = os.path.join(base_dir, subpath)
    
    if os.path.exists(file_path):
        return send_from_directory(base_dir, subpath)
    else:
        return "File not found.", 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
