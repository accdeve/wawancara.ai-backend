import cv2
from flask import Flask, request, jsonify, render_template
import tempfile
import os

# Import your detection functions
from gaze_detection import detect_gaze
from hand_detection import detect_hands
from head_detection import detect_head
from smile_detection import detect_smiles

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 3000 * 1024 * 1024

@app.route("/")
def hello():
    return render_template("index.html")

@app.route('/detection', methods=['POST'])
def detect_movements_endpoint():
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400
    
    key = request.args.get('key')
    if key != 'pkm-dikti':
        return jsonify({"status": "error", "message": "Invalid key provided"}), 400

    video_file = request.files['video']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    try:
        smile_count = detect_smiles(temp_video_path)
        hand_count = detect_hands(temp_video_path)
        head_count = detect_head(temp_video_path)
        gaze_count = detect_gaze(temp_video_path)
        
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    finally:
        os.remove(temp_video_path)

    return jsonify({
        "status": "success",
        "message": "Detection completed",
        "data": {
            "smile_count": smile_count,
            "hand_count": hand_count,
            "head_count": head_count,
            "gaze_count": gaze_count
        }
    }), 200

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
