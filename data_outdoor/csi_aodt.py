import socket
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import multiprocessing as mp
from collections import deque
import os
import time

# --- Config ---
LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5000
OUTPUT_DIR = "/home/inss/phyan/data"
SAVE_EVERY_N = 1000
NUM_WORKERS = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Shared state for plotting (main process only, protected by data_lock) ---
data_lock = threading.Lock()
latest_csi_data = {}
ta_history = deque(maxlen=2000)
rx_ports = set()

# --- Matplotlib globals ---
fig = None
axs = {}
lines = {}


def parse_packet(data):
    """Parse one raw UDP payload into a structured dict, or None if invalid."""
    total_len = len(data)
    if total_len < 16:
        return None

    float_region_len = total_len - 16
    float_count = float_region_len // 4
    floats = np.frombuffer(data[:float_region_len], dtype=np.float32)

    if float_count < 2:
        return None

    rx_port = int(floats[0])
    tx_port = int(floats[1])

    time_alignment_s, timestamp = struct.unpack(
        "<dQ", data[float_region_len:float_region_len + 16]
    )

    if float_count > 2:
        csi_float = floats[2:].reshape(-1, 2)
        csi = csi_float[:, 0] + 1j * csi_float[:, 1]
    else:
        csi = np.array([])

    return {
        "rx_port": rx_port,
        "tx_port": tx_port,
        "ta_us": time_alignment_s * 1e6,
        "timestamp": timestamp,
        "csi": csi,
    }


def parse_worker(raw_queue, result_queue):
    """Separate process: pulls raw UDP payloads and parses them (CPU-bound work)."""
    while True:
        data = raw_queue.get()
        if data is None:  # shutdown sentinel
            break
        parsed = parse_packet(data)
        if parsed is not None:
            result_queue.put(parsed)


def saver_process(save_queue, output_dir, save_every_n):
    """Separate process: buffers parsed entries and writes them out as .npz files."""
    buffer = []
    saved_file_count = 0

    def flush():
        nonlocal saved_file_count
        if not buffer:
            return
        timestamp_now = int(time.time() * 1000)
        filename = os.path.join(
            output_dir,
            f"csi_snapshot_{timestamp_now}_{saved_file_count + 1}.npz"
        )
        np.savez_compressed(
            filename,
            rx_port=np.array([d["rx_port"] for d in buffer]),
            tx_port=np.array([d["tx_port"] for d in buffer]),
            ta_us=np.array([d["ta_us"] for d in buffer]),
            csi=np.array([d["csi"] for d in buffer], dtype=object),
            timestamp=np.array([d["timestamp"] for d in buffer]),
        )
        saved_file_count += 1
        print(f"[Saved] {filename}, {len(buffer)} entries, saved_file_count={saved_file_count}")
        buffer.clear()

    while True:
        item = save_queue.get()
        if item is None:  # shutdown sentinel
            flush()
            break
        buffer.append(item)
        if len(buffer) >= save_every_n:
            flush()


def udp_receiver(raw_queue, listen_ip, listen_port):
    """Thread in the main process: only recv + hand off, stays fast so no packets are dropped."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((listen_ip, listen_port))
    print(f"Listening(UDP) on {listen_ip}:{listen_port} ...")

    counter = 0
    while True:
        data, addr = sock.recvfrom(8192)
        if not data:
            continue
        raw_queue.put(bytes(data))
        counter += 1
        if counter % 200 == 0:
            print(f"[recv] {counter} packets received")


def result_consumer(result_queue, save_queue):
    """Thread in the main process: updates live-plot state and forwards entries to the saver."""
    counter = 0
    while True:
        parsed = result_queue.get()
        if parsed is None:  # shutdown sentinel
            break

        with data_lock:
            latest_csi_data[parsed["rx_port"]] = parsed["csi"]
            ta_history.append(parsed["ta_us"])
            rx_ports.add(parsed["rx_port"])

        save_queue.put(parsed)

        counter += 1
        if counter % 200 == 0:
            print(
                f"[{counter}] RX={parsed['rx_port']}, TX={parsed['tx_port']}, "
                f"TA={parsed['ta_us']:.3f} us, CSI_len={len(parsed['csi'])}, "
                f"timestamp={parsed['timestamp']}"
            )


def setup_plots():
    global fig, axs, lines

    while not rx_ports:
        threading.Event().wait(0.5)

    n_rx = len(rx_ports)
    rx_port_list = sorted(list(rx_ports))

    print(f"Setting up plots for RX ports: {rx_port_list}")

    fig = plt.figure(figsize=(4 * (n_rx + 1), 8))
    gs = fig.add_gridspec(2, n_rx + 1)

    for i, rx_port in enumerate(rx_port_list):
        ax_mag = fig.add_subplot(gs[0, i])
        l1, = ax_mag.plot([], [], label=f"RX {rx_port}")
        ax_mag.set_title(f"RX {rx_port} - Magnitude", fontsize=14)
        ax_mag.set_ylabel("Magnitude", fontsize=12)
        ax_mag.set_xlabel("Subcarrier Index", fontsize=12)
        ax_mag.grid(True)
        ax_mag.set_ylim(0, 1)

        ax_phase = fig.add_subplot(gs[1, i])
        l2, = ax_phase.plot([], [], label=f"RX {rx_port}")
        ax_phase.set_title(f"RX {rx_port} - Phase", fontsize=14)
        ax_phase.set_ylabel("Unwrapped Phase (rad)", fontsize=12)
        ax_phase.set_xlabel("Subcarrier Index", fontsize=12)
        ax_phase.grid(True)
        ax_phase.set_ylim(-10, 10)

        axs[rx_port] = {"mag": ax_mag, "phase": ax_phase}
        lines[rx_port] = {"mag": l1, "phase": l2}

    ax_ta = fig.add_subplot(gs[:, n_rx])
    l_ta, = ax_ta.plot([], [], "r.-", label="Time Alignment")
    ax_ta.set_title("Time Alignment History", fontsize=14)
    ax_ta.set_ylabel("Time Alignment (us)", fontsize=12)
    ax_ta.set_xlabel("Sample Index", fontsize=12)
    ax_ta.grid(True)
    ax_ta.legend()

    axs["ta"] = ax_ta
    lines["ta"] = l_ta

    plt.tight_layout(pad=2.0)


def update_plots(frame):
    updated_lines = []

    with data_lock:
        local_csi_data = latest_csi_data.copy()
        local_ta_history = list(ta_history)

    if not local_csi_data or not lines:
        return []

    for rx_port, csi_data in local_csi_data.items():
        if rx_port not in lines:
            continue

        mag = np.abs(csi_data)
        phase_unwrapped = np.unwrap(np.angle(csi_data))
        x_data = np.arange(len(mag))

        lines[rx_port]["mag"].set_data(x_data, mag)
        axs[rx_port]["mag"].set_xlim(0, len(mag) - 1 if len(mag) > 1 else 1)

        lines[rx_port]["phase"].set_data(x_data, phase_unwrapped)
        axs[rx_port]["phase"].set_xlim(0, len(phase_unwrapped) - 1 if len(phase_unwrapped) > 1 else 1)

        updated_lines.extend([lines[rx_port]["mag"], lines[rx_port]["phase"]])

    if "ta" in lines and local_ta_history:
        ta_line = lines["ta"]
        ta_ax = axs["ta"]

        x_ta_data = np.arange(len(local_ta_history))
        ta_line.set_data(x_ta_data, local_ta_history)

        ta_ax.relim()
        ta_ax.autoscale_view(True, True, True)

        updated_lines.append(ta_line)

    return updated_lines


if __name__ == "__main__":
    print(f"Using {NUM_WORKERS} worker process(es) for parsing.")
    print(f"Saving 1 file for every {SAVE_EVERY_N} entries.")
    print(f"Output directory: {OUTPUT_DIR}")

    raw_queue = mp.Queue()
    result_queue = mp.Queue()
    save_queue = mp.Queue()

    workers = [
        mp.Process(target=parse_worker, args=(raw_queue, result_queue), daemon=True)
        for _ in range(NUM_WORKERS)
    ]
    for w in workers:
        w.start()

    saver = mp.Process(target=saver_process, args=(save_queue, OUTPUT_DIR, SAVE_EVERY_N), daemon=True)
    saver.start()

    recv_thread = threading.Thread(target=udp_receiver, args=(raw_queue, LISTEN_IP, LISTEN_PORT), daemon=True)
    recv_thread.start()

    consumer_thread = threading.Thread(target=result_consumer, args=(result_queue, save_queue), daemon=True)
    consumer_thread.start()

    try:
        setup_plots()
        ani = FuncAnimation(fig, update_plots, interval=50, blit=False)
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        for _ in workers:
            raw_queue.put(None)
        save_queue.put(None)
        for w in workers:
            w.join(timeout=2)
        saver.join(timeout=5)
