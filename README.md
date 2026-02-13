# S.A.R.A. Vision Control System

Real-time face tracking with MQTT servo control.

## Run Instructions

### 1. Backend (VPS)
```bash
cd backend
npm install
node server.js
```

### 2. Vision (PC)
```bash
python -m src.enroll  # First time only
python src/vision_node.py --broker 157.173.101.159 --name YOUR_NAME
```

### 3. ESP8266 (Arduino)
- Open `esp8266/vision_servo/vision_servo.ino`
- Update WiFi credentials
- Upload to board

### 4. Dashboard
Open: `http://157.173.101.159:8080`

---

**Team 313**
