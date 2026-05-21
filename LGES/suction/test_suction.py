import requests
import time


MODBUS_HOST = "192.168.1.1"
MODBUS_PORT = 502



def stop_processes():
    requests.post("http://192.168.1.1/api/dc/weblogic/stop")
    print("Stopped processes.")

def set_suction_1():
    stop_processes()
    time.sleep(0.5)
    requests.post("http://192.168.1.1/api/dc/weblogic/run/3587")
    print("Suction = 1")

def set_suction_0():
    stop_processes()
    time.sleep(0.5)
    requests.post("http://192.168.1.1/api/dc/weblogic/run/763")
    print("Suction = 0")

def set_blow_1():
    stop_processes()
    time.sleep(0.5)
    requests.post("http://192.168.1.1/api/dc/weblogic/run/7381")
    print("Blow = 1")

def set_blow_0():
    stop_processes()
    time.sleep(0.5)
    requests.post("http://192.168.1.1/api/dc/weblogic/run/5484")
    print("Blow = 0")

if __name__ == "__main__":
    try:
        set_suction_0()
        while True:
            print("  s = Suction ON  |  d = Suction OFF")
            print("  z = Blow ON     |  x = Blow OFF")
            print("  q = Quit")
            print("Input command :", end=" ")
            command = input().strip().lower()

            if command == "s":
                set_suction_1()
            elif command == "d":
                set_suction_0()
            elif command == "z":
                set_blow_1()
            elif command == "x":
                set_blow_0()
            elif command == "q":
                break
    finally:
        print("Resetting to default state...")
        set_suction_0()
        set_blow_0()