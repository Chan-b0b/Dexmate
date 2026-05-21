import customtkinter as ctk
import requests
import threading
import time

BASE_URL = "http://192.168.1.1/api/dc/weblogic"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# --- API functions ---

def stop_processes():
    requests.post(f"{BASE_URL}/stop")

def set_suction_1():
    stop_processes()
    time.sleep(0.5)
    requests.post(f"{BASE_URL}/run/3587")

def set_suction_0():
    stop_processes()
    time.sleep(0.5)
    requests.post(f"{BASE_URL}/run/763")

def set_blow_1():
    stop_processes()
    time.sleep(0.5)
    requests.post(f"{BASE_URL}/run/7381")

def set_blow_0():
    stop_processes()
    time.sleep(0.5)
    requests.post(f"{BASE_URL}/run/5484")


# --- GUI ---

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Process Controller")
        self.geometry("420x520")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._busy = False

        self._build_ui()
        self._run(set_suction_0, "Initialised — Suction OFF")
        self._run(set_blow_0, "Initialised — Blow OFF")

    # ------------------------------------------------------------------ UI build

    def _build_ui(self):
        padx = 20

        # Title
        ctk.CTkLabel(self, text="Process Controller",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(padx=padx, pady=(24, 4))

        # Status badge
        self._status_var = ctk.StringVar(value="Ready")
        ctk.CTkLabel(self, textvariable=self._status_var,
                     font=ctk.CTkFont(size=12),
                     text_color="gray70").pack()

        # Suction section
        suction_frame = ctk.CTkFrame(self, corner_radius=12)
        suction_frame.pack(fill="x", padx=padx, pady=(18, 6))

        ctk.CTkLabel(suction_frame, text="Suction",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(12, 8))

        btn_row = ctk.CTkFrame(suction_frame, fg_color="transparent")
        btn_row.pack(pady=(0, 14))

        self._s1_btn = ctk.CTkButton(btn_row, text="ON", width=140, height=44,
                                     fg_color="#1f6aa5", hover_color="#174f7a",
                                     font=ctk.CTkFont(size=14, weight="bold"),
                                     command=lambda: self._run(set_suction_1, "Suction ON"))
        self._s1_btn.grid(row=0, column=0, padx=8)

        self._s0_btn = ctk.CTkButton(btn_row, text="OFF", width=140, height=44,
                                     fg_color="#555555", hover_color="#3a3a3a",
                                     font=ctk.CTkFont(size=14, weight="bold"),
                                     command=lambda: self._run(set_suction_0, "Suction OFF"))
        self._s0_btn.grid(row=0, column=1, padx=8)

        # Blow section
        blow_frame = ctk.CTkFrame(self, corner_radius=12)
        blow_frame.pack(fill="x", padx=padx, pady=6)

        ctk.CTkLabel(blow_frame, text="Blow",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(12, 8))

        btn_row2 = ctk.CTkFrame(blow_frame, fg_color="transparent")
        btn_row2.pack(pady=(0, 14))

        self._b1_btn = ctk.CTkButton(btn_row2, text="ON", width=140, height=44,
                                     fg_color="#1f6aa5", hover_color="#174f7a",
                                     font=ctk.CTkFont(size=14, weight="bold"),
                                     command=lambda: self._run(set_blow_1, "Blow ON"))
        self._b1_btn.grid(row=0, column=0, padx=8)

        self._b0_btn = ctk.CTkButton(btn_row2, text="OFF", width=140, height=44,
                                     fg_color="#555555", hover_color="#3a3a3a",
                                     font=ctk.CTkFont(size=14, weight="bold"),
                                     command=lambda: self._run(set_blow_0, "Blow OFF"))
        self._b0_btn.grid(row=0, column=1, padx=8)

        # Log box
        self._log = ctk.CTkTextbox(self, height=110, corner_radius=10,
                                   font=ctk.CTkFont(family="Courier", size=12),
                                   state="disabled")
        self._log.pack(fill="x", padx=padx, pady=(14, 6))

        # Quit button
        ctk.CTkButton(self, text="Quit", width=160, height=40,
                      fg_color="#8b0000", hover_color="#5c0000",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      command=self._on_close).pack(pady=(6, 20))

    # ------------------------------------------------------------------ helpers

    def _log_msg(self, msg: str):
        self._log.configure(state="normal")
        self._log.insert("end", f"» {msg}\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _set_busy(self, busy: bool):
        self._busy = busy
        state = "disabled" if busy else "normal"
        for btn in (self._s1_btn, self._s0_btn, self._b1_btn, self._b0_btn):
            btn.configure(state=state)
        self._status_var.set("Working…" if busy else "Ready")

    def _run(self, func, message: str = None):
        if self._busy:
            return

        def task():
            self.after(0, self._set_busy, True)
            try:
                func()
                if message:
                    self.after(0, self._log_msg, message)
            except Exception as e:
                self.after(0, self._log_msg, f"Error: {e}")
            finally:
                self.after(0, self._set_busy, False)

        threading.Thread(target=task, daemon=True).start()

    def _on_close(self):
        self._log_msg("Resetting to safe state before exit…")
        try:
            set_suction_0()
            set_blow_0()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
