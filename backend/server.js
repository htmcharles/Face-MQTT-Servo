const mqtt = require('mqtt');
const WebSocket = require('ws');
const http = require('http');
const fs = require('fs');
const path = require('path');

// Configuration
const MQTT_BROKER = 'mqtt://127.0.0.1';
const TEAM_ID = 'year3d/team7';
const MQTT_TOPIC_VS = `vision/${TEAM_ID}/movement`;
const BACKEND_PORT = 8322;
const FRONTEND_PORT = 9322;
const DASHBOARD_PATH = path.join(__dirname, '../dashboard/index.html');

// --- MQTT Client ---
console.log(`Connecting to MQTT Broker: ${MQTT_BROKER}...`);
const mqttClient = mqtt.connect(MQTT_BROKER);

mqttClient.on('connect', () => {
    console.log(`Connected to MQTT Broker.`);
    mqttClient.subscribe(MQTT_TOPIC_VS, (err) => {
        if (!err) {
            console.log(`Subscribed to topic: ${MQTT_TOPIC_VS}`);
        } else {
            console.error('MQTT Subscription Error:', err);
        }
    });
});

mqttClient.on('message', (topic, message) => {
    const msgString = message.toString();
    console.log(`MQTT IN [${topic}]: ${msgString}`);

    // Broadcast to all WS clients
    broadcast(msgString);
});

// --- HTTP Server for Dashboard ---
const server = http.createServer((req, res) => {
    if (req.url === '/' || req.url === '/index.html') {
        fs.readFile(DASHBOARD_PATH, (err, data) => {
            if (err) {
                console.error('Error loading dashboard:', err);
                res.writeHead(500);
                res.end('Error loading dashboard: ' + err.message);
                return;
            }
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(data);
        });
    } else {
        res.writeHead(404);
        res.end('Not Found');
    }
});

// --- WebSocket Server (Standalone on backend port) ---
const wss = new WebSocket.Server({ port: BACKEND_PORT });

console.log(`HTTP Server (Frontend) started on port ${FRONTEND_PORT}`);
console.log(`WebSocket Server (Backend) started on port ${BACKEND_PORT}`);

wss.on('connection', (ws) => {
    console.log('New WebSocket Client connected');

    // Send initial status
    ws.send(JSON.stringify({ type: 'STATUS', message: 'Connected to Vision Backend' }));

    ws.on('close', () => {
        console.log('Client disconnected');
    });
});

function broadcast(data) {
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(data);
        }
    });
}

// Start HTTP server
server.listen(FRONTEND_PORT, () => {
    console.log(`Frontend ready. Dashboard available at http://localhost:${FRONTEND_PORT}`);
});
