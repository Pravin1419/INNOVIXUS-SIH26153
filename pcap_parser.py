from scapy.all import rdpcap, IP, TCP, UDP
from pathlib import Path
import pandas as pd
import sys


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

OUTPUT_FILE = Path(r".\data\packet_features.csv")


# --------------------------------------------------
# CHECK INPUT
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage:")
    print("python pcap_parser.py <path_to_pcap>")
    sys.exit(1)

PCAP_FILE = Path(sys.argv[1])

if not PCAP_FILE.exists():
    print(f"ERROR: PCAP file not found: {PCAP_FILE}")
    sys.exit(1)


# --------------------------------------------------
# LOAD PCAP
# --------------------------------------------------

print("=" * 60)
print("INNOVIXUS - PCAP PACKET PARSER")
print("=" * 60)

print("\nPCAP:", PCAP_FILE)
print("Loading packets...")

packets = rdpcap(str(PCAP_FILE))

print("Packets loaded:", len(packets))


# --------------------------------------------------
# PACKET FEATURE EXTRACTION
# --------------------------------------------------

records = []

for packet in packets:

    # We need an IP packet
    if not packet.haslayer(IP):
        continue

    ip = packet[IP]

    timestamp = float(packet.time)

    ttl = int(ip.ttl)

    packet_length = len(packet)

    payload_size = 0

    if packet.haslayer(TCP):
        transport = packet[TCP]

        payload_size = len(bytes(transport.payload))

        tcp_window = int(transport.window)

        syn = int(bool(transport.flags & 0x02))
        ack = int(bool(transport.flags & 0x10))
        fin = int(bool(transport.flags & 0x01))
        rst = int(bool(transport.flags & 0x04))
        psh = int(bool(transport.flags & 0x08))
        urg = int(bool(transport.flags & 0x20))

    elif packet.haslayer(UDP):

        transport = packet[UDP]

        payload_size = len(bytes(transport.payload))

        tcp_window = 0

        syn = 0
        ack = 0
        fin = 0
        rst = 0
        psh = 0
        urg = 0

    else:

        tcp_window = 0

        syn = 0
        ack = 0
        fin = 0
        rst = 0
        psh = 0
        urg = 0


    # IP fragmentation
    fragment_offset = int(ip.frag)

    more_fragments = int(
        bool(ip.flags & 0x01)
    )


    records.append({

        "Timestamp": timestamp,

        "TTL": ttl,

        "Packet_Length": packet_length,

        "Payload_Size": payload_size,

        "TCP_Window": tcp_window,

        "SYN": syn,

        "ACK": ack,

        "FIN": fin,

        "RST": rst,

        "PSH": psh,

        "URG": urg,

        "Fragment_Offset": fragment_offset,

        "More_Fragments": more_fragments
    })


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(records)


if df.empty:

    print("\nERROR: No IP packets found in PCAP.")

    sys.exit(1)


# --------------------------------------------------
# TIMESTAMP
# --------------------------------------------------

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    unit="s"
)


# --------------------------------------------------
# RETRANSMISSION APPROXIMATION
# --------------------------------------------------

df["Retransmission"] = 0

previous_sequences = {}

for index, packet in enumerate(packets):

    if not packet.haslayer(IP):
        continue

    if not packet.haslayer(TCP):
        continue

    ip = packet[IP]
    tcp = packet[TCP]

    connection = (
        ip.src,
        ip.dst,
        tcp.sport,
        tcp.dport
    )

    sequence = int(tcp.seq)

    if connection in previous_sequences:

        if sequence <= previous_sequences[connection]:

            df.loc[index, "Retransmission"] = 1

    previous_sequences[connection] = sequence


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("PACKET FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print("\nExtracted packets :", len(df))
print("Features          :", len(df.columns))

print("\nFeatures:")

for column in df.columns:
    print(" -", column)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFirst 5 packets:")
print(df.head().to_string(index=False))