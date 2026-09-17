import socket

HOST = '0.0.0.0'  # или '0.0.0.0' чтобы слушать все интерфейсы
PORT = 7012

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))
print(f"Слушаю UDP {HOST}:{PORT} ... (Ctrl+C для выхода)")

try:
    while True:
        data, addr = sock.recvfrom(65536)
        text = data.decode('utf-8', errors='replace')
        print(f"[{addr[0]}:{addr[1]}] {text}")
        print()
except KeyboardInterrupt:
    print("\nОстановлено")
finally:
    sock.close()