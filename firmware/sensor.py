import struct
import time
import network
import socket
import json
from machine import I2C, Pin, PWM

# I2C setup
i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)

# Wake up MPU-6050
i2c.writeto_mem(104, 0x6B, b'\x00')

# Set sample rate to 1000Hz
i2c.writeto_mem(104, 0x19, b'\x00')

def read_accel():
    raw = i2c.readfrom_mem(104, 0x3B, 6)
    x, y, z = struct.unpack('>3h', raw)
    return z / 16384.0  # use Z axis, convert to g

def collect_samples(n=500):
    samples = []
    for _ in range(n):
        try:
            samples.append(read_accel())
        except:
            samples.append(0.0)
        time.sleep_us(2000)  # 2ms = 500Hz instead of 1000Hz
    return samples

# WiFi connection
sta = network.WLAN(network.STA_IF)
sta.active(True)
if not sta.isconnected():
    import wifi
    wifi.connect_wifi('L231 WiFi', '9175970278')

print('Sensor ready. IP:', sta.ifconfig()[0])

# Motor control - auto start
in1 = Pin(14, Pin.OUT)
in2 = Pin(27, Pin.OUT)
ena = PWM(Pin(26), freq=1000)
in1.value(1)
in2.value(0)
ena.duty(1023)
print('Motor spinning!')

# Main loop
while True:
    try:
        samples = collect_samples(500)
        data = json.dumps({'samples': samples})
        
        s = socket.socket()
        s.settimeout(5)
        addr = socket.getaddrinfo('192.168.1.22', 5001)[0][-1]
        s.connect(addr)
        
        # Send in chunks
        encoded = data.encode()
        s.sendall(encoded)
        s.close()
        print('Sent', len(samples), 'samples')
        
    except Exception as e:
        print('Error:', e)
        try:
            s.close()
        except:
            pass
        time.sleep(2)
    
    time.sleep(1)