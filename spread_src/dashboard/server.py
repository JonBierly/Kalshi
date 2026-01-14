from flask import Flask, jsonify, send_from_directory
import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

app = Flask(__name__, static_folder='static')

DATA_PATH = 'data/dashboard_state.json'

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/state')
def get_state():
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "Dashboard state not found. Make sure rebalancing_live_trader.py is running."}), 404
    
    try:
        with open(DATA_PATH, 'r') as f:
            state = json.load(f)
        return jsonify(state)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    print("=" * 60)
    print("NBA SPREAD TRADING DASHBOARD")
    print("=" * 60)
    print("Go to: http://127.0.0.1:5050")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5050, debug=True)
