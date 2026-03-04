"""
FastAPI server with Socket.IO for real-time PPE detection
"""

import asyncio
import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import socketio
import json
from typing import Dict, Any

# Initialize FastAPI and Socket.IO
app = FastAPI(title="PPE Detection API")
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True
)

# Mount Socket.IO to FastAPI
asgi_app = socketio.ASGIApp(sio, app)

# Load model
try:
    model = YOLO("https://huggingface.co/yihong1120/Construction-Hazard-Detection/resolve/main/models/yolo11/pt/yolo11n.pt")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Detection settings
ALLOWED_CLASSES = {0, 1, 2, 3, 4, 5, 7}  # Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Vest
CONFIDENCE_THRESHOLD = 0.25

# Store active connections
active_connections: Dict[str, WebSocket] = {}

# HTML template for real-time detection
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PPE Detection - Real-time</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .video-container { position: relative; display: inline-block; }
        #video { width: 640px; height: 480px; border: 2px solid #333; }
        .controls { margin: 20px 0; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; }
        .start-btn { background: #4CAF50; color: white; border: none; }
        .stop-btn { background: #f44336; color: white; border: none; }
        .detection-info { margin: 20px 0; padding: 15px; background: white; border-radius: 5px; }
        .detection-item { display: inline-block; margin: 5px; padding: 5px 10px; background: #e3f2fd; border-radius: 3px; }
        .person-box { border: 2px solid #2196F3; }
        .ppe-log { margin: 10px 0; padding: 10px; background: #fff3cd; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚧 PPE Detection - Real-time</h1>
        
        <div class="video-container">
            <video id="video" autoplay playsinline></video>
            <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        </div>
        
        <div class="controls">
            <button id="startBtn" class="start-btn" onclick="startDetection()">📹 Start Detection</button>
            <button id="stopBtn" class="stop-btn" onclick="stopDetection()" disabled>⏹ Stop Detection</button>
        </div>
        
        <div class="detection-info">
            <h3>📊 Detection Results</h3>
            <div id="personCount">Persons: 0</div>
            <div id="ppeLog" class="ppe-log">PPE Items: None detected</div>
        </div>
    </div>

    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <script>
        let socket = io();
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let ctx = canvas.getContext('2d');
        let isDetecting = false;
        let stream = null;

        socket.on('connect', () => {
            console.log('Connected to server');
        });

        socket.on('detection_result', (data) => {
            console.log('Detection result:', data);
            
            // Update person count
            document.getElementById('personCount').textContent = `Persons: ${data.person_count}`;
            
            // Update PPE log
            if (data.ppe_items.length > 0) {
                const ppeText = data.ppe_items.map(item => 
                    `${item.class}(${item.confidence.toFixed(2)})`
                ).join(', ');
                document.getElementById('ppeLog').innerHTML = `PPE Items: ${ppeText}`;
            } else {
                document.getElementById('ppeLog').innerHTML = 'PPE Items: None detected';
            }
            
            // Draw bounding boxes on canvas
            if (data.image) {
                const img = new Image();
                img.onload = () => {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    
                    // Draw person boxes
                    if (data.person_boxes) {
                        ctx.strokeStyle = '#2196F3';
                        ctx.lineWidth = 2;
                        data.person_boxes.forEach(box => {
                            ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
                            ctx.fillStyle = '#2196F3';
                            ctx.fillRect(box.x1, box.y1 - 25, 60, 25);
                            ctx.fillStyle = 'white';
                            ctx.font = '16px Arial';
                            ctx.fillText('Person', box.x1 + 5, box.y1 - 8);
                        });
                    }
                };
                img.src = `data:image/jpeg;base64,${data.image}`;
            }
        });

        socket.on('error', (data) => {
            console.error('Server error:', data);
            alert('Detection error: ' + data.message);
        });

        async function startDetection() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 } 
                });
                
                video.srcObject = stream;
                video.play();
                
                isDetecting = true;
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                
                // Start sending frames
                sendFrame();
                
            } catch (error) {
                console.error('Error accessing camera:', error);
                alert('Error accessing camera: ' + error.message);
            }
        }

        function stopDetection() {
            isDetecting = false;
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }

        function sendFrame() {
            if (!isDetecting) return;
            
            // Capture frame from video
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send frame to server
            socket.emit('frame', { image: imageData.split(',')[1] });
            
            // Continue sending frames
            requestAnimationFrame(sendFrame);
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    """Serve the real-time detection page"""
    return HTMLResponse(content=HTML_TEMPLATE)

@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    print(f"Client connected: {sid}")
    active_connections[sid] = None  # Will store WebSocket when needed

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    print(f"Client disconnected: {sid}")
    if sid in active_connections:
        del active_connections[sid]

@sio.event
async def frame(sid, data):
    """Process frame from client"""
    try:
        if not model:
            await sio.emit('error', {'message': 'Model not loaded'}, room=sid)
            return
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            await sio.emit('error', {'message': 'Invalid frame data'}, room=sid)
            return
        
        # Run detection
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        
        # Process detections
        person_boxes = []
        ppe_items = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)
                
                # Only show bounding boxes for persons
                if class_id == 5:  # Person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    })
                
                # Log PPE items
                if class_id in [0, 1, 2, 3, 4, 7]:  # PPE classes
                    ppe_items.append({
                        'class': class_name,
                        'confidence': confidence
                    })
        
        # Encode frame for response
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send results back to client
        await sio.emit('detection_result', {
            'person_count': len(person_boxes),
            'person_boxes': person_boxes,
            'ppe_items': ppe_items,
            'image': frame_base64
        }, room=sid)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting FastAPI server with Socket.IO...")
    print("📹 Real-time PPE Detection Server")
    print("🌐 Open http://localhost:8000 for real-time detection")
    
    uvicorn.run(
        asgi_app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
