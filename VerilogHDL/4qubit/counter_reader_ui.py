import csv
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


DEFAULT_BAUD = 500000
SYNC_WAIT_SEC = 1.25

NUM_CELL = 256
SIGNAL_BIN_WIDTH = 16
IDLER_BIN_WIDTH = 16
COIN_BIN_WIDTH = 16
CELL_BIN_WIDTH = 16
MULTI2_WIDTH = 6
MULTI3_WIDTH = 4
MULTI4_WIDTH = 2
READ_TIMES_WIDTH = 40
WRITE_TIMES_WIDTH = 40

MULTI2_OFFSET = 0
MULTI3_OFFSET = MULTI2_OFFSET + MULTI2_WIDTH
MULTI4_OFFSET = MULTI3_OFFSET + MULTI3_WIDTH
SIGNAL_HIST_OFFSET = MULTI4_OFFSET + MULTI4_WIDTH
IDLER_HIST_OFFSET = SIGNAL_HIST_OFFSET + NUM_CELL * SIGNAL_BIN_WIDTH
COIN_HIST_OFFSET = IDLER_HIST_OFFSET + NUM_CELL * IDLER_BIN_WIDTH
READ_TIMES_OFFSET = COIN_HIST_OFFSET + NUM_CELL * COIN_BIN_WIDTH
WRITE_TIMES_OFFSET = READ_TIMES_OFFSET + READ_TIMES_WIDTH
SIGNAL_CELL_OFFSET = WRITE_TIMES_OFFSET + WRITE_TIMES_WIDTH
IDLER_CELL_OFFSET = SIGNAL_CELL_OFFSET + 10 * CELL_BIN_WIDTH
COUNTER_USED_BITS = IDLER_CELL_OFFSET + 10 * CELL_BIN_WIDTH
DEFAULT_FRAME_BYTES = (COUNTER_USED_BITS + 7) // 8


def get_bits(value, start, width):
    return (value >> start) & ((1 << width) - 1)


def parse_counter_frame(frame):
    value = int.from_bytes(frame, byteorder="big", signed=False)

    signal_hist = [
        get_bits(value, SIGNAL_HIST_OFFSET + i * SIGNAL_BIN_WIDTH, SIGNAL_BIN_WIDTH)
        for i in range(NUM_CELL)
    ]
    idler_hist = [
        get_bits(value, IDLER_HIST_OFFSET + i * IDLER_BIN_WIDTH, IDLER_BIN_WIDTH)
        for i in range(NUM_CELL)
    ]
    coin_hist = [
        get_bits(value, COIN_HIST_OFFSET + i * COIN_BIN_WIDTH, COIN_BIN_WIDTH)
        for i in range(NUM_CELL)
    ]
    signal_cellhist = [
        get_bits(value, SIGNAL_CELL_OFFSET + i * CELL_BIN_WIDTH, CELL_BIN_WIDTH)
        for i in range(10)
    ]
    idler_cellhist = [
        get_bits(value, IDLER_CELL_OFFSET + i * CELL_BIN_WIDTH, CELL_BIN_WIDTH)
        for i in range(10)
    ]

    return {
        "multi_2": get_bits(value, MULTI2_OFFSET, MULTI2_WIDTH),
        "multi_3": get_bits(value, MULTI3_OFFSET, MULTI3_WIDTH),
        "multi_4": get_bits(value, MULTI4_OFFSET, MULTI4_WIDTH),
        "read_times": get_bits(value, READ_TIMES_OFFSET, READ_TIMES_WIDTH),
        "write_times": get_bits(value, WRITE_TIMES_OFFSET, WRITE_TIMES_WIDTH),
        "signal_hist": signal_hist,
        "idler_hist": idler_hist,
        "coin_hist": coin_hist,
        "signal_cellhist": signal_cellhist,
        "idler_cellhist": idler_cellhist,
        "raw": frame,
    }


def read_exact(ser, size, stop_event):
    data = bytearray()
    while len(data) < size and not stop_event.is_set():
        chunk = ser.read(size - len(data))
        if chunk:
            data.extend(chunk)
    return bytes(data)


class BarCanvas(ttk.Frame):
    def __init__(self, master, title):
        super().__init__(master)
        self.title = title
        self.values = []
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

    def set_values(self, values):
        self.values = list(values)
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)

        margin_left = 44
        margin_right = 12
        margin_top = 22
        margin_bottom = 28
        plot_w = max(width - margin_left - margin_right, 1)
        plot_h = max(height - margin_top - margin_bottom, 1)

        self.canvas.create_text(
            margin_left,
            8,
            anchor="nw",
            text=self.title,
            fill="#222222",
            font=("Segoe UI", 10, "bold"),
        )

        if not self.values:
            self.canvas.create_text(
                width // 2,
                height // 2,
                text="No frame loaded",
                fill="#777777",
                font=("Segoe UI", 10),
            )
            return

        max_value = max(self.values) or 1
        bar_count = len(self.values)
        bar_w = max(plot_w / bar_count, 1)

        self.canvas.create_line(
            margin_left,
            margin_top,
            margin_left,
            margin_top + plot_h,
            fill="#666666",
        )
        self.canvas.create_line(
            margin_left,
            margin_top + plot_h,
            margin_left + plot_w,
            margin_top + plot_h,
            fill="#666666",
        )
        self.canvas.create_text(
            margin_left - 8,
            margin_top,
            anchor="e",
            text=str(max_value),
            fill="#555555",
            font=("Segoe UI", 8),
        )
        self.canvas.create_text(
            margin_left - 8,
            margin_top + plot_h,
            anchor="e",
            text="0",
            fill="#555555",
            font=("Segoe UI", 8),
        )

        for i, value in enumerate(self.values):
            x0 = margin_left + i * bar_w
            x1 = margin_left + (i + 1) * bar_w - 1
            y1 = margin_top + plot_h
            y0 = y1 - (value / max_value) * plot_h
            color = "#2563eb" if bar_count > 32 else "#0f766e"
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

        if bar_count > 16:
            for label in (0, bar_count // 2, bar_count - 1):
                x = margin_left + label * bar_w
                self.canvas.create_text(
                    x,
                    margin_top + plot_h + 14,
                    text=str(label),
                    fill="#555555",
                    font=("Segoe UI", 8),
                )
        else:
            for i in range(bar_count):
                x = margin_left + i * bar_w + bar_w / 2
                self.canvas.create_text(
                    x,
                    margin_top + plot_h + 14,
                    text=str(i),
                    fill="#555555",
                    font=("Segoe UI", 8),
                )


class CounterReaderUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FPGA Counter Reader")
        self.geometry("1040x720")
        self.minsize(900, 600)

        self.ser = None
        self.worker = None
        self.stop_event = threading.Event()
        self.messages = queue.Queue()
        self.last_data = None

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUD))
        self.frame_var = tk.StringVar(value=str(DEFAULT_FRAME_BYTES))
        self.auto_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Disconnected")

        self._build_ui()
        self.refresh_ports()
        self.after(100, self._poll_messages)

        if serial is None:
            messagebox.showwarning(
                "Missing dependency",
                "pyserial is not installed. Run: pip install pyserial",
            )

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(root)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Port").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(controls, textvariable=self.port_var, width=18)
        self.port_combo.grid(row=0, column=1, padx=(6, 12), sticky="w")
        ttk.Button(controls, text="Refresh", command=self.refresh_ports).grid(row=0, column=2, padx=(0, 12))

        ttk.Label(controls, text="Baud").grid(row=0, column=3, sticky="w")
        ttk.Entry(controls, textvariable=self.baud_var, width=10).grid(row=0, column=4, padx=(6, 12), sticky="w")

        ttk.Label(controls, text="Frame bytes").grid(row=0, column=5, sticky="w")
        ttk.Entry(controls, textvariable=self.frame_var, width=8).grid(row=0, column=6, padx=(6, 12), sticky="w")

        self.connect_btn = ttk.Button(controls, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=7, padx=(0, 8))
        ttk.Button(controls, text="Read synced", command=self.read_once).grid(row=0, column=8, padx=(0, 8))
        ttk.Checkbutton(controls, text="Auto", variable=self.auto_var, command=self.toggle_auto).grid(row=0, column=9)
        ttk.Button(controls, text="Save CSV", command=self.save_csv).grid(row=0, column=10, padx=(8, 0))

        status = ttk.Label(root, textvariable=self.status_var, anchor="w")
        status.pack(fill=tk.X, pady=(8, 8))

        body = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body, padding=(0, 0, 12, 0))
        body.add(left, weight=1)

        self.summary = tk.Text(left, width=34, height=18, wrap="word")
        self.summary.pack(fill=tk.BOTH, expand=True)
        self.summary.configure(state="disabled")

        right = ttk.Notebook(body)
        body.add(right, weight=4)

        self.signal_plot = BarCanvas(right, "Signal hist")
        self.idler_plot = BarCanvas(right, "Idler hist")
        self.coin_plot = BarCanvas(right, "Coin hist")
        self.cell_plot = BarCanvas(right, "Cell hist: signal 0-9, idler 10-19")

        right.add(self.signal_plot, text="Signal")
        right.add(self.idler_plot, text="Idler")
        right.add(self.coin_plot, text="Coin")
        right.add(self.cell_plot, text="Cell")

    def refresh_ports(self):
        if list_ports is None:
            self.port_combo["values"] = []
            return
        ports = [port.device for port in list_ports.comports()]
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])

    def toggle_connection(self):
        if self.ser and self.ser.is_open:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        if serial is None:
            messagebox.showerror("Missing dependency", "Install pyserial first: pip install pyserial")
            return False

        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("No port", "Choose a serial port first.")
            return False

        try:
            baud = int(self.baud_var.get())
            self.ser = serial.Serial(port, baudrate=baud, timeout=0.2, write_timeout=0.2)
            self.ser.reset_input_buffer()
            self.connect_btn.configure(text="Disconnect")
            self.status_var.set(f"Connected to {port} @ {baud}")
            return True
        except Exception as exc:
            messagebox.showerror("Connect failed", str(exc))
            self.ser = None
            return False

    def disconnect(self):
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.0)
        self.worker = None
        self.stop_event.clear()

        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.auto_var.set(False)
        self.connect_btn.configure(text="Connect")
        self.status_var.set("Disconnected")

    def frame_size(self):
        return int(self.frame_var.get())

    def read_once(self):
        if not self.ser or not self.ser.is_open:
            if not self.connect():
                return
        self._start_worker(auto=False)

    def toggle_auto(self):
        if self.auto_var.get():
            if not self.ser or not self.ser.is_open:
                if not self.connect():
                    self.auto_var.set(False)
                    return
            self._start_worker(auto=True)
        else:
            self.stop_event.set()

    def _start_worker(self, auto):
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._reader_worker, args=(auto,), daemon=True)
        self.worker.start()

    def _sync_stream(self):
        self.ser.write(b"n")
        self.ser.flush()
        time.sleep(SYNC_WAIT_SEC)
        self.ser.reset_input_buffer()
        self.ser.write(b"y")
        self.ser.flush()

    def _reader_worker(self, auto):
        try:
            frame_size = self.frame_size()
            self.messages.put(("status", "Synchronizing frame..."))
            self._sync_stream()

            while not self.stop_event.is_set():
                frame = read_exact(self.ser, frame_size, self.stop_event)
                if len(frame) != frame_size:
                    self.messages.put(("status", f"Read stopped: got {len(frame)} / {frame_size} bytes"))
                    break
                parsed = parse_counter_frame(frame)
                self.messages.put(("frame", parsed))
                if not auto:
                    break
        except Exception as exc:
            self.messages.put(("error", str(exc)))
        finally:
            if not auto:
                self.messages.put(("auto", False))

    def _poll_messages(self):
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break

            if kind == "frame":
                self.last_data = payload
                self._render_data(payload)
            elif kind == "status":
                self.status_var.set(payload)
            elif kind == "error":
                self.status_var.set(f"Error: {payload}")
                messagebox.showerror("Serial error", payload)
            elif kind == "auto":
                self.auto_var.set(bool(payload))

        self.after(100, self._poll_messages)

    def _render_data(self, data):
        lines = [
            f"Frame bytes: {len(data['raw'])}",
            f"Write times: {data['write_times']}",
            f"Read times:  {data['read_times']}",
            "",
            f"2-click count: {data['multi_2']}",
            f"3-click count: {data['multi_3']}",
            f"4-click count: {data['multi_4']}",
            "",
            f"Signal total: {sum(data['signal_hist'])}",
            f"Idler total:  {sum(data['idler_hist'])}",
            f"Coin total:   {sum(data['coin_hist'])}",
            "",
            "Signal cellhist:",
            " ".join(str(v) for v in data["signal_cellhist"]),
            "",
            "Idler cellhist:",
            " ".join(str(v) for v in data["idler_cellhist"]),
        ]
        self.summary.configure(state="normal")
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, "\n".join(lines))
        self.summary.configure(state="disabled")

        self.signal_plot.set_values(data["signal_hist"])
        self.idler_plot.set_values(data["idler_hist"])
        self.coin_plot.set_values(data["coin_hist"])
        self.cell_plot.set_values(data["signal_cellhist"] + data["idler_cellhist"])
        self.status_var.set("Frame loaded")

    def save_csv(self):
        if not self.last_data:
            messagebox.showinfo("No data", "Read a frame before saving.")
            return

        path = filedialog.asksaveasfilename(
            title="Save counter CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        data = self.last_data
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["field", "index", "value"])
            writer.writerow(["write_times", "", data["write_times"]])
            writer.writerow(["read_times", "", data["read_times"]])
            writer.writerow(["multi_2", "", data["multi_2"]])
            writer.writerow(["multi_3", "", data["multi_3"]])
            writer.writerow(["multi_4", "", data["multi_4"]])
            for name in ("signal_hist", "idler_hist", "coin_hist", "signal_cellhist", "idler_cellhist"):
                for index, value in enumerate(data[name]):
                    writer.writerow([name, index, value])

        self.status_var.set(f"Saved {path}")


if __name__ == "__main__":
    app = CounterReaderUI()
    app.mainloop()
