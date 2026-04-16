import network
import time

def connect_wifi(ssid, password):
    sta = network.WLAN(network.STA_IF)
    sta.active(False)
    time.sleep(2)
    sta.active(True)
    time.sleep(1)
    if not sta.isconnected():
        print('Connecting to WiFi...')
        sta.connect(ssid, password)
        timeout = 0
        while not sta.isconnected() and timeout < 30:
            time.sleep(1)
            timeout += 1
    if sta.isconnected():
        print('Connected! IP:', sta.ifconfig()[0])
        return True
    else:
        print('Failed to connect')
        return False