    
    
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches

class VirtualInstrument(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phần mềm mô phỏng hệ đo bức xạ")
        self.geometry("1250x850")
        self.configure(bg="#2b2b2b") 

        # --- BIẾN VẬT LÝ ---
        self.true_rate = 100.0          # Tốc độ phát xung (cps)
        self.tau = 0.005                # Thời gian chết mặc định (5ms)
        self.target_pulses = 10         
        self.window_size = self.target_pulses / self.true_rate 

        self.is_running = False
        self.time_now = 0.0
        self.next_event_time = np.random.exponential(1.0 / self.true_rate)

        self.events_real, self.events_non, self.events_par = [], [], []
        self.blocks_non, self.blocks_par = [], []
        self.count_real, self.count_non, self.count_par = 0, 0, 0
        self.last_dead_non, self.last_arrival_par = -self.tau, -self.tau
        
        self.free_time_non = 0.0
        self.free_time_par = 0.0
        self.dead_time_pct = 0.0

        self.setup_ui()
        self.setup_plot()
        self.update_simulation()

    def reset_data(self):
        self.time_now = 0.0
        self.next_event_time = np.random.exponential(1.0 / self.true_rate)
        self.events_real, self.events_non, self.events_par = [], [], []
        self.blocks_non, self.blocks_par = [], []
        self.count_real, self.count_non, self.count_par = 0, 0, 0
        self.last_dead_non, self.last_arrival_par = -self.tau, -self.tau
        self.free_time_non, self.free_time_par = 0.0, 0.0
        self.dead_time_pct = 0.0
        
        self.update_displays()

    def setup_ui(self):
        led_font = ("Courier", 24, "bold")
        label_font = ("Arial", 11, "bold")
        
        control_frame = tk.Frame(self, bg="#3c3f41", width=380, bd=5, relief="sunken")
        control_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        # ================= MODULE 1: MÁY PHÁT XUNG =================
        gen_frame = tk.LabelFrame(control_frame, text=" MÁY PHÁT XUNG NGẪU NHIÊN ", bg="#3c3f41", fg="#a9b7c6", font=label_font)
        gen_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)
        
        self.disp_rate = tk.Label(gen_frame, text=f"{int(self.true_rate)} cps", font=led_font, bg="black", fg="#00ff00", bd=3, relief="sunken", pady=5)
        self.disp_rate.pack(fill="x", padx=10, pady=5)
        
        input_frame = tk.Frame(gen_frame, bg="#3c3f41")
        input_frame.pack(fill="x", padx=10, pady=(5,0))
        tk.Label(input_frame, text="Tốc độ (cps):", bg="#3c3f41", fg="white").pack(side="left")
        
        self.rate_str = tk.StringVar(value=str(int(self.true_rate)))
        self.rate_entry = tk.Entry(input_frame, textvariable=self.rate_str, width=8, font=("Arial", 10, "bold"), justify="center")
        self.rate_entry.pack(side="left", padx=10)
        self.rate_entry.bind('<Return>', self.on_rate_entry)
        self.rate_entry.bind('<FocusOut>', self.on_rate_entry)
        tk.Label(input_frame, text="(Max: 50000)", bg="#3c3f41", fg="#888888", font=("Arial", 9)).pack(side="left")

        # Cập nhật thanh trượt từ 1 đến 3000
        self.rate_slider = ttk.Scale(gen_frame, from_=1, to=50000, orient="horizontal", command=self.on_rate_slider)
        self.rate_slider.set(self.true_rate)
        self.rate_slider.pack(fill="x", padx=10, pady=(5, 10))

        # ================= MODULE 2: MÁY ĐẾM XUNG =================
        count_frame = tk.LabelFrame(control_frame, text=" MÁY ĐẾM BỨC XẠ ", bg="#3c3f41", fg="#a9b7c6", font=label_font)
        count_frame.pack(fill="x", padx=10, pady=5, ipadx=5, ipady=5)
        
        led_frame = tk.Frame(count_frame, bg="black", bd=3, relief="sunken")
        led_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Label(led_frame, text="TIME (s)", bg="black", fg="#00aa00", font=("Arial", 8)).grid(row=0, column=0, sticky="w", padx=5)
        self.disp_time = tk.Label(led_frame, text="0.00", font=led_font, bg="black", fg="#00ff00")
        self.disp_time.grid(row=1, column=0, sticky="w", padx=5)
        
        tk.Label(led_frame, text="COUNTS", bg="black", fg="#00aa00", font=("Arial", 8)).grid(row=0, column=1, sticky="w", padx=20)
        self.disp_count = tk.Label(led_frame, text="0", font=led_font, bg="black", fg="#00ff00")
        self.disp_count.grid(row=1, column=1, sticky="w", padx=20)

        tk.Label(led_frame, text="DEAD TIME (%)", bg="black", fg="#aa0000", font=("Arial", 8)).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=(10,0))
        self.disp_dt_pct = tk.Label(led_frame, text="0.0 %", font=led_font, bg="black", fg="#ff3333")
        self.disp_dt_pct.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=(0,5))
        
        # Khung nhập thời gian chết (Tau)
        tau_input_frame = tk.Frame(count_frame, bg="#3c3f41")
        tau_input_frame.pack(fill="x", padx=10, pady=(5,0))
        tk.Label(tau_input_frame, text="Thời gian xử lý \u03C4 (ms):", bg="#3c3f41", fg="white").pack(side="left")
        
        self.tau_str = tk.StringVar(value=f"{self.tau*1000:.3f}")
        self.tau_entry = tk.Entry(tau_input_frame, textvariable=self.tau_str, width=8, font=("Arial", 10, "bold"), justify="center")
        self.tau_entry.pack(side="right")
        self.tau_entry.bind('<Return>', self.on_tau_entry)
        self.tau_entry.bind('<FocusOut>', self.on_tau_entry)
        
        self.tau_slider = ttk.Scale(count_frame, from_=0.001, to=300.0, orient="horizontal", command=self.on_tau_change)
        self.tau_slider.set(self.tau * 1000)
        self.tau_slider.pack(fill="x", padx=10, pady=5)
        
        self.mode_var = tk.StringVar(value="non")
        ttk.Radiobutton(count_frame, text="Cố định (Non-paralyzable)", variable=self.mode_var, value="non", command=self.force_redraw).pack(anchor="w", padx=10, pady=2)
        ttk.Radiobutton(count_frame, text="Liệt (Paralyzable)", variable=self.mode_var, value="par", command=self.force_redraw).pack(anchor="w", padx=10, pady=2)

        btn_frame = tk.Frame(count_frame, bg="#3c3f41")
        btn_frame.pack(fill="x", padx=10, pady=10)
        self.btn_toggle = tk.Button(btn_frame, text="START / STOP", font=label_font, bg="#2b8c5a", fg="white", command=self.toggle_state, height=2)
        self.btn_toggle.pack(side="left", expand=True, fill="x", padx=(0,5))
        tk.Button(btn_frame, text="RESET", font=label_font, bg="#c94a4a", fg="white", command=self.reset_data, height=2).pack(side="right", expand=True, fill="x", padx=(5,0))

        # ================= MODULE 3: CÀI ĐẶT ĐỒ THỊ =================
        graph_ctrl_frame = tk.LabelFrame(control_frame, text=" CÀI ĐẶT HIỂN THỊ ĐỒ THỊ ", bg="#3c3f41", fg="#a9b7c6", font=label_font)
        graph_ctrl_frame.pack(fill="x", padx=10, pady=5, ipadx=5, ipady=5)
        
        win_frame = tk.Frame(graph_ctrl_frame, bg="#3c3f41")
        win_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(win_frame, text="Số xung hiển thị \ntrên khung đồ thị:", bg="#3c3f41", fg="white", justify="left").pack(side="left")
        
        self.pulses_str = tk.StringVar(value=str(self.target_pulses))
        self.pulses_entry = tk.Entry(win_frame, textvariable=self.pulses_str, width=6, font=("Arial", 10, "bold"), justify="center")
        self.pulses_entry.pack(side="right", padx=10)
        self.pulses_entry.bind('<Return>', self.update_window_size)
        self.pulses_entry.bind('<FocusOut>', self.update_window_size)

    def setup_plot(self):
        graph_frame = tk.Frame(self, bg="#2b2b2b")
        graph_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        self.fig.patch.set_facecolor('#2b2b2b')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- CÁC HÀM XỬ LÝ SỰ KIỆN GIAO DIỆN ---
    def on_rate_slider(self, val):
        self.true_rate = float(val)
        self.rate_str.set(str(int(self.true_rate)))
        self.disp_rate.config(text=f"{int(self.true_rate)} cps")
        self.calc_window_size()

    def on_rate_entry(self, event=None):
        try:
            val = float(self.rate_str.get())
            val = max(1, min(50000, val)) # Giới hạn từ 1 đến 3000 cps
            self.true_rate = val
            self.rate_slider.set(val)
            self.rate_str.set(str(int(val)))
            self.disp_rate.config(text=f"{int(self.true_rate)} cps")
            self.calc_window_size()
        except ValueError:
            self.rate_str.set(str(int(self.true_rate))) 

    def on_tau_change(self, val):
        v = float(val)
        self.tau = v / 1000.0
        self.tau_str.set(f"{v:.1f}")

    def on_tau_entry(self, event=None):
        try:
            val = float(self.tau_str.get())
            val = max(0.001, min(300.0, val)) # Giới hạn từ 0.1ms đến 300ms
            self.tau = val / 1000.0
            self.tau_slider.set(val)
            self.tau_str.set(f"{val:.1f}")
        except ValueError:
            self.tau_str.set(f"{self.tau*1000:.1f}")

    def update_window_size(self, event=None):
        try:
            val = int(self.pulses_str.get())
            self.target_pulses = max(1, val) 
            self.pulses_str.set(str(self.target_pulses))
            self.calc_window_size()
        except ValueError:
            self.pulses_str.set(str(self.target_pulses))

    def calc_window_size(self):
        if self.true_rate > 0:
            self.window_size = self.target_pulses / self.true_rate

    def toggle_state(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_toggle.config(bg="#c94a4a", text="STOP")
        else:
            self.btn_toggle.config(bg="#2b8c5a", text="START")

    def force_redraw(self):
        if not self.is_running:
            self.draw_graphs()
            self.update_displays()

    # --- LOGIC MÔ PHỎNG VÀ VẼ ĐỒ THỊ ---
    def update_simulation(self):
        if self.is_running:
            dt = 0.005 
            self.time_now += dt

            while self.next_event_time < self.time_now:
                t = self.next_event_time
                self.count_real += 1
                self.events_real.append(t)
                
                if t >= self.last_dead_non + self.tau:
                    self.free_time_non += t - max(0, self.last_dead_non + self.tau)
                    self.count_non += 1
                    self.events_non.append(t)
                    self.blocks_non.append((t, t + self.tau))
                    self.last_dead_non = t
                    
                if t >= self.last_arrival_par + self.tau:
                    self.free_time_par += t - max(0, self.last_arrival_par + self.tau)
                    self.count_par += 1
                    self.events_par.append(t)
                    
                self.blocks_par.append((t, t + self.tau)) 
                self.last_arrival_par = t
                
                self.next_event_time += np.random.exponential(1.0 / self.true_rate)

            if self.time_now > 0:
                if self.mode_var.get() == "non":
                    cur_free = self.free_time_non
                    if self.time_now > self.last_dead_non + self.tau:
                        cur_free += self.time_now - max(0, self.last_dead_non + self.tau)
                else:
                    cur_free = self.free_time_par
                    if self.time_now > self.last_arrival_par + self.tau:
                        cur_free += self.time_now - max(0, self.last_arrival_par + self.tau)
                        
                dead_time = self.time_now - cur_free
                self.dead_time_pct = (dead_time / self.time_now) * 100
            
            if int(self.time_now * 1000) % 50 < 10: 
                self.draw_graphs()
                self.update_displays()

        self.after(5, self.update_simulation)

    def draw_graphs(self):
        self.ax1.clear()
        self.ax2.clear()

        min_t = max(0, self.time_now - self.window_size)
        max_t = max(self.window_size, self.time_now)

        ev_real_plot = [e for e in self.events_real if min_t <= e <= max_t]
        self.ax1.vlines(ev_real_plot, 0, 1, color='cyan', lw=2)
        self.ax1.set_title(f"XUNG TỚI ĐẦU DÒ (Đã phát sinh {self.count_real} hạt)", color='white', fontsize=10)
        self.ax1.set_ylim(0, 1.2)
        self.ax1.set_yticks([])
        self.ax1.grid(color='#444444', linestyle='--', alpha=0.5)

        mode = self.mode_var.get()
        self.ax2.vlines(ev_real_plot, 0, 0.4, color='gray', linestyle=':', label='Xung tới')

        if mode == "non":
            ev_plot = [e for e in self.events_non if min_t <= e <= max_t]
            blocks = self.blocks_non
            title = "CỬA SỔ THỜI GIAN CHẾT (KHÔNG LIỆT)"
            color = 'orange'
        else:
            ev_plot = [e for e in self.events_par if min_t <= e <= max_t]
            blocks = self.blocks_par
            title = "CỬA SỔ THỜI GIAN CHẾT (LIỆT)"
            color = 'red'

        self.ax2.vlines(ev_plot, 0, 1, color='#00ff00', lw=2, label='Xung ghi nhận')
        for start, end in [b for b in blocks if b[1] >= min_t and b[0] <= max_t]:
            self.ax2.add_patch(patches.Rectangle((start, 0), self.tau, 1, color=color, alpha=0.4))

        self.ax2.set_title(title, color='white', fontsize=10)
        self.ax2.set_xlim(min_t, max_t)
        self.ax2.set_ylim(0, 1.2)
        self.ax2.set_yticks([])
        self.ax2.set_xlabel("Thời gian trôi (s)", color='white')
        self.ax2.grid(color='#444444', linestyle='--', alpha=0.5)
        
        self.ax2.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white')

        self.fig.tight_layout()
        self.canvas.draw()

    def update_displays(self):
        self.disp_time.config(text=f"{self.time_now:.2f}")
        count = self.count_non if self.mode_var.get() == "non" else self.count_par
        self.disp_count.config(text=f"{count}")
        self.disp_dt_pct.config(text=f"{self.dead_time_pct:.1f} %")

if __name__ == "__main__":
    app = VirtualInstrument()
    app.mainloop()