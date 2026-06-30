import socket

HOST = '192.168.50.101'
PORT = 23

def scan_barcode():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        try:
            s.connect((HOST,PORT))
            print("Connected to Congent scanner.")
            # Send trigger command (Native Mode: 'se8' for In-Sight, 'TRIGGER ON' for DataMan)
            command = b"T\r\n"
            s.sendall(command)
            
            # Read the response from the scanner
            data = s.recv(1024)
            response = data.decode('utf-8').strip()
            
            print(f"Scanner Response: {response}")
            return response
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    scan_barcode()
