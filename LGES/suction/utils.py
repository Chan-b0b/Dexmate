def listen_data_in():
    sio = socketio.Client()
    last_data_in = {}

    @sio.on("*")
    def catch_all(event, data):
        nonlocal last_data_in
        try:
            current = data["computebox"]["variable"]["dInput"][0]
            if current != last_data_in:
                print(f"DI0 = {current}" + (" Suction Completed!" if current == 1 else " Suction Released."))
                last_data_in = current
        except (KeyError, TypeError):
            pass

    sio.connect(
        "http://192.168.1.1",
        transports=["websocket", "polling"],
        socketio_path="socket.io",
    )
    sio.wait()