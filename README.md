# Vision Control System

A real-time face tracking and recognition system designed for distributed robotic control. It uses arcface for identity verification and MediaPipe for facial expression and movement detection.

## System Overview
The system captures video from a local camera, identifies enrolled users, and tracks their facial movements. Movement commands (LEFT, RIGHT, CENTERED) are published via MQTT to a central broker, which then broadcasts them to an ESP8266-controlled servo and a web-based monitoring dashboard.

## MQTT Topics
| Topic | Description |
| --- | --- |
| `vision/year3d/team7/movement` | Movement commands, lock status, and face snapshots. |
| `vision/year3d/team7/heartbeat` | Periodic node status updates (ONLINE/OFFLINE). |

## Run Instructions

### 1. Backend (VPS)
```bash
cd backend
npm install
node server.js
```

### 2. Vision (PC)
```bash
python -m src.enroll  # First time only to enroll your face
python src/vision_node.py --name Your_Name
```

### 3. ESP8266 (Arduino)
- Open `esp8266/vision_servo/vision_servo.ino`
- Update WiFi credentials and MQTT Broker IP (`157.173.101.159`)
- Upload to board

### 4. Live Dashboard
The dashboard is served by the backend and provides real-time monitoring and manual controls.
- **URL**: [http://157.173.101.159:9322](http://157.173.101.159:9322)

---

**Team Year3D - Team 7**
