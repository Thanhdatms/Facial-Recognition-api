import time
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion

# Configuration
broker = "192.168.208.211"  # Địa chỉ broker của bạn
port = 1883                 # Cổng mặc định của MQTT
topic = "esp32/data"        # Topic để gửi tin nhắn
message = "Hello, MQTT!"    # Tin nhắn mẫu

# Initialize MQTT client with Callback API Version 2 and MQTT v5
client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)

# Callback functions
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected to MQTT broker successfully")
        # Gửi tin nhắn ngay khi kết nối thành công
        send_message(client, topic, message)
    else:
        print(f"Failed to connect to MQTT broker with code: {reason_code}")

def on_disconnect(client, userdata, flags, reason_code, properties):
    print(f"Disconnected from MQTT broker with code: {reason_code}")

def on_publish(client, userdata, mid, reason_code, properties):
    print(f"Message {mid} published successfully")

# Set callbacks
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_publish = on_publish

def send_message(client, topic, message):
    """Gửi tin nhắn MQTT nếu đã kết nối."""
    if client.is_connected():
        result = client.publish(topic, message, qos=1)
        print(f"Sending message '{message}' to topic '{topic}'")
    else:
        print("MQTT client is not connected, cannot send message")

def connect_mqtt():
    """Kết nối đến broker và xử lý thử lại nếu thất bại."""
    try:
        client.connect(broker, port)
        client.loop_start()  # Bắt đầu vòng lặp xử lý MQTT trong luồng nền
    except Exception as e:
        print(f"Error connecting to MQTT broker: {e}")
        print("Retrying in 5 seconds...")
        time.sleep(5)
        connect_mqtt()  # Thử kết nối lại

if __name__ == "__main__":
    # Kết nối đến broker
    connect_mqtt()

    # Vòng lặp chính để giữ chương trình chạy
    try:
        while True:
            # Kiểm tra trạng thái kết nối và gửi tin nhắn mỗi 10 giây
            if client.is_connected():
                print("MQTT is connected, sending periodic message...")
                send_message(client, topic, "Periodic message")
            else:
                print("MQTT is not connected, attempting to reconnect...")
                connect_mqtt()
            time.sleep(10)
    except KeyboardInterrupt:
        print("Stopping program...")
        client.loop_stop()
        client.disconnect()