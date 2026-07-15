import threading
import time

import requests
import socketio

MODBUS_HOST = "192.168.5.1"
MODBUS_PORT = 502


class SealMonitor:
    """Watches DI0 (vacuum seal input) + toolA over the controller's socketio stream.

    DI0 goes T the moment a vacuum seal is achieved while suction is ON, and
    stays T after suction OFF until the cup physically releases (see test_di0.py).
    """

    def __init__(self, host=MODBUS_HOST):
        self._sio = socketio.Client()
        self._host = host
        self.connected = False
        self.di0 = False
        self.tool_a = 0.0

        @self._sio.on("*")
        def _on_data(event, data):
            try:
                var = data["computebox"]["variable"]
            except (KeyError, TypeError):
                return
            try:
                self.di0 = bool(var["dInput"][0])
            except (KeyError, TypeError, IndexError):
                pass
            try:
                self.tool_a = float(var["toolA"])
            except (KeyError, TypeError, ValueError):
                pass

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            self._sio.connect(
                f"http://{self._host}",
                transports=["websocket", "polling"],
                socketio_path="socket.io",
            )
            self.connected = True
            self._sio.wait()
        except Exception as exc:
            print(f"Seal monitor connection failed: {exc}")


def check_seal(monitor):
    if not monitor.connected:
        print("Seal monitor not connected — DI0 unavailable.")
        return
    state = "T  -> SUCTIONED (seal)" if monitor.di0 else "F  -> no seal"
    print(f"DI0 = {state}   |   toolA = {monitor.tool_a:.4f} A")


def stop_processes():
    requests.post(f"http://{MODBUS_HOST}/api/dc/weblogic/stop")
    print("Stopped processes.")

def set_suction_1():
    stop_processes()
    time.sleep(0.5)
    requests.post(f"http://{MODBUS_HOST}/api/dc/weblogic/run/3587")
    print("Suction = 1")

def set_suction_0():
    stop_processes()
    time.sleep(0.5)
    requests.post(f"http://{MODBUS_HOST}/api/dc/weblogic/run/763")
    print("Suction = 0")

def set_blow_1():
    stop_processes()
    time.sleep(0.5)
    requests.post(f"http://{MODBUS_HOST}/api/dc/weblogic/run/7381")
    print("Blow = 1")

def set_blow_0():
    stop_processes()
    time.sleep(0.5)
    requests.post(f"http://{MODBUS_HOST}/api/dc/weblogic/run/5484")
    print("Blow = 0")

if __name__ == "__main__":
    monitor = SealMonitor()
    monitor.start()
    try:
        set_suction_0()
        while True:
            check_seal(monitor)
            print("  s = Suction ON  |  d = Suction OFF")
            print("  z = Blow ON     |  x = Blow OFF")
            print("  c = Check seal (DI0)")
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
            elif command == "c":
                check_seal(monitor)
            elif command == "q":
                break
    finally:
        print("Resetting to default state...")
        set_suction_0()
        set_blow_0()