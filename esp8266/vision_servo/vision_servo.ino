#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

// --- Configuration ---
// WiFi Credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// MQTT Broker (VPS IP)
const char* mqtt_server = "157.173.101.159"; 

const int mqtt_port = 1883;
const char* client_id = "esp8266_year3d_team7";
const char* topic_movement = "vision/year3d/team7/movement";
const char* topic_heartbeat = "vision/year3d/team7/heartbeat";

// Servo Configuration
Servo myServo;
const int servoPin =  GPIO14 ; // GPIO14 (Common on NodeMCU)
int currentAngle = 90;   // Start at center

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");

  // Convert payload to string
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.println(message);

  // Simple parsing (JSON parsing in C++ is tedious without a library, using string matching)
  if (message.indexOf("MOVE_LEFT") > 0) {
    moveServo(5); // Move Left
  } else if (message.indexOf("MOVE_RIGHT") > 0) {
    moveServo(-5); // Move Right
  } else if (message.indexOf("CENTERED") > 0) {
    // Optional: Hold position or recenter
  }
}

void moveServo(int delta) {
  currentAngle += delta;
  if (currentAngle < 0) currentAngle = 0;
  if (currentAngle > 180) currentAngle = 180;
  
  myServo.write(currentAngle);
  Serial.print("Servo Angle: ");
  Serial.println(currentAngle);
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Attempt to connect
    if (client.connect(client_id)) {
      Serial.println("connected");
      // Subscribe
      client.subscribe(topic_movement);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  
  // Setup Servo
  myServo.attach(servoPin);
  myServo.write(currentAngle); // Center

  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // Heartbeat every 5 seconds
  static unsigned long lastMsg = 0;
  unsigned long now = millis();
  if (now - lastMsg > 5000) {
    lastMsg = now;
    String heartbeat = "{\"node\": \"esp8266\", \"status\": \"ONLINE\"}";
    client.publish(topic_heartbeat, heartbeat.c_str());
  }
}
