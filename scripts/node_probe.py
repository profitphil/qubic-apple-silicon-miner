#!/usr/bin/env python3
"""Read-only Qubic node probe: connect, read greeting, request current tick info.
Protocol: 8-byte header = size(3B LE) | type(1B) | dejavu(4B LE), then payload.
REQUEST_CURRENT_TICK_INFO = 27 -> RESPOND_CURRENT_TICK_INFO = 28 (CurrentTickInfo).
EXCHANGE_PUBLIC_PEERS = 0 is typically sent by the node on connect.
"""
import socket, struct, sys, random, time

HOST = sys.argv[1] if len(sys.argv) > 1 else "51.77.52.28"
PORT = 21841

def header(size, msg_type, dejavu):
    return struct.pack("<I", size & 0xFFFFFF | (msg_type << 24)) + struct.pack("<I", dejavu)

def parse_header(b):
    v = struct.unpack("<I", b[:4])[0]
    return v & 0xFFFFFF, v >> 24, struct.unpack("<I", b[4:8])[0]  # size, type, dejavu

s = socket.create_connection((HOST, PORT), timeout=8)
s.settimeout(8)
print(f"[+] connected to {HOST}:{PORT}")

# request current tick info
dejavu = random.randint(1, 0x7FFFFFFF)
req = header(8, 27, dejavu)
s.sendall(req)
print(f"[+] sent REQUEST_CURRENT_TICK_INFO (type 27, dejavu {dejavu:#x})")

deadline = time.time() + 10
buf = b""
found = False
while time.time() < deadline and not found:
    try:
        chunk = s.recv(65536)
    except socket.timeout:
        break
    if not chunk:
        break
    buf += chunk
    while len(buf) >= 8:
        size, mtype, dj = parse_header(buf)
        if size < 8 or size > 0x1000000:
            print(f"[!] bad header size {size}; resyncing")
            buf = buf[1:]
            continue
        if len(buf) < size:
            break
        payload = buf[8:size]
        buf = buf[size:]
        if mtype == 0:
            n = len(payload) // 4
            peers = [".".join(str(b) for b in payload[i*4:(i+1)*4]) for i in range(min(n, 4))]
            print(f"[<] EXCHANGE_PUBLIC_PEERS: {n} peers, first: {peers}")
        elif mtype == 28 and len(payload) >= 16:
            tickDuration, epoch = struct.unpack("<HH", payload[0:4])
            tick, nAligned, nMisaligned = struct.unpack("<IHH", payload[4:12])
            initialTick = struct.unpack("<I", payload[12:16])[0]
            print(f"[<] CURRENT_TICK_INFO: epoch={epoch} tick={tick} "
                  f"tickDuration={tickDuration}ms initialTick={initialTick} "
                  f"votes={nAligned}/{nAligned+nMisaligned}")
            found = True
        else:
            print(f"[<] message type={mtype} size={size} (payload {len(payload)}B) "
                  f"head={payload[:16].hex()}")
s.close()
print("[+] done" if found else "[!] no tick info received (node may gate requests)")
