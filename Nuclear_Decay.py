import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv

# Global variables
ani = None
times, simulated_counts, theoretical_counts, diff_log = [], [], [], []
stop_requested = False
current_frame = 0
frames = 0
alive = None
positions = None
lambda_decay = None
N0 = 0
dt = 0
total_time = 0
scatter = None
count_text = None
line_sim = None
line_theory = None
line_diff = None
ax_curve = None
ax_diff = None
use_fixed_seed = True
hide_decayed_var = None
rel_diff_log = []      # Danh sách lưu sai khác tương đối
ax_rel_diff = None     # Trục (axes) cho đồ thị mới
line_rel_diff = None   # Đường (line) cho đồ thị mới

def update(frame):
    global current_frame, stop_requested, alive, times, simulated_counts, theoretical_counts, diff_log
    global line_rel_diff, rel_diff_log 
    
    current_frame = frame

    if stop_requested:
        ani.event_source.stop()
        status_label.config(text="Đã dừng mô phỏng.")
        return scatter, line_sim, line_theory, line_diff, line_rel_diff, count_text

    if frame >= frames:
        ani.event_source.stop()
        status_label.config(text="Hoàn tất mô phỏng.")
        return scatter, line_sim, line_theory, line_diff, line_rel_diff, count_text

    if frame == 0:
        return scatter, line_sim, line_theory, line_diff, line_rel_diff, count_text

    t = frame * dt
    p = 1 - np.exp(-lambda_decay * dt)
    decay_events = (np.random.rand(N0) < p) & alive
    alive[decay_events] = False

    # (Code ẩn/hiện hạt giữ nguyên)
    colors = np.zeros((N0, 4))
    alive_mask = alive
    colors[alive_mask] = [1, 0, 0, 1.0] 
    decayed_mask = ~alive
    colors[decayed_mask] = [0.5, 0.5, 0.5, 0.3]
    hide = hide_decayed_var.get()
    if hide:
        scatter.set_offsets(positions[alive_mask])
        scatter.set_facecolor(colors[alive_mask])
    else:
        scatter.set_offsets(positions)
        scatter.set_facecolor(colors)
    
    count_text.set_text(f"Số hạt: {alive.sum()}")

    times.append(t)
    sim_count = alive.sum()
    simulated_counts.append(sim_count)
    theory_count = N0 * np.exp(-lambda_decay * t)
    theoretical_counts.append(theory_count)
    
    abs_diff = sim_count - theory_count
    diff_log.append(abs_diff)
    rel_diff = (abs_diff / (theory_count + 1e-9)) * 100 
    rel_diff_log.append(rel_diff)

    line_sim.set_data(times, simulated_counts)
    line_theory.set_data(times, theoretical_counts)
    ax_curve.set_xlim(0, total_time)
    ax_curve.set_ylim(0.1 if scale_var.get() == 'log' else 0, N0)

    # --- SỬA LOGIC TRỤC Y CỦA SAI KHÁC TUYỆT ĐỐI ---
    line_diff.set_data(times, diff_log)
    ax_diff.set_xlim(0, total_time)
    if len(diff_log) > 10: # Cần đủ dữ liệu để tính percentile
        ymin = np.percentile(diff_log, 5)  # Phân vị 5%
        ymax = np.percentile(diff_log, 95) # Phân vị 95%
        padding = (ymax - ymin) * 0.1 # Thêm 10% đệm
        if padding < 1:
            padding = 1 # Đệm tối thiểu
        
        ax_diff.set_ylim(ymin - padding, ymax + padding)
    # --- KẾT THÚC SỬA ---

    # --- SỬA LOGIC TRỤC Y CỦA SAI KHÁC TƯƠNG ĐỐI ---
    line_rel_diff.set_data(times, rel_diff_log)
    ax_rel_diff.set_xlim(0, total_time)
    if len(rel_diff_log) > 10:
        # Bỏ qua vài điểm đầu tiên vì (N -> 0) gây đột biến
        data_to_analyze = rel_diff_log[5:] 
        if data_to_analyze:
            ymin = np.percentile(data_to_analyze, 5)
            ymax = np.percentile(data_to_analyze, 95)
            padding = (ymax - ymin) * 0.1
            if padding < 1:
                padding = 1
            
            # Giới hạn giá trị để tránh vô cực
            ymin = max(-200, ymin - padding) 
            ymax = min(200, ymax + padding)  
            
            ax_rel_diff.set_ylim(ymin, ymax)
    # --- KẾT THÚC SỬA ---

    return scatter, line_sim, line_theory, line_diff, line_rel_diff, count_text
    
def start_simulation():
    global ani, times, simulated_counts, theoretical_counts, diff_log
    global stop_requested, current_frame, frames, alive, positions, lambda_decay, N0, dt, total_time
    global scatter, count_text, line_sim, line_theory, line_diff, ax_curve, ax_diff
    # --- SỬA DÒNG GLOBAL ---
    global ax_rel_diff, line_rel_diff, rel_diff_log
    # --- KẾT THÚC SỬA ---

    if ani is not None:
        try:
            ani.event_source.stop()
        except AttributeError:
            pass
        ani = None

    times.clear()
    simulated_counts.clear()
    theoretical_counts.clear()
    diff_log.clear()
    # --- THÊM ---
    rel_diff_log.clear() 
    # --- KẾT THÚC THÊM ---
    
    stop_requested = False
    current_frame = 0

    try:
        N0 = int(entry_particles.get())
        half_life = float(entry_half_life.get())
        total_time = float(entry_total_time.get())
        dt = float(entry_time_step.get())
    except ValueError:
        status_label.config(text="Vui lòng nhập giá trị số hợp lệ.")
        return

    lambda_decay = np.log(2) / half_life

    np.random.seed(42 if use_fixed_seed else None)

    positions = np.random.rand(N0, 2)
    alive = np.ones(N0, dtype=bool)

    fig.clf()
    
    # --- SỬA LAYOUT (TỪ 1x3 THÀNH 2x2) ---
    ax_anim = fig.add_subplot(2, 2, 1)  # (hàng 2, cột 2, vị trí 1)
    ax_curve = fig.add_subplot(2, 2, 2) # (vị trí 2)
    ax_diff = fig.add_subplot(2, 2, 3)  # (vị trí 3)
    ax_rel_diff = fig.add_subplot(2, 2, 4) # (vị trí 4)
    # --- KẾT THÚC SỬA ---

    # (Phần code ax_anim giữ nguyên)
    ax_anim.set_xlim(0, 1)
    ax_anim.set_ylim(0, 1)
    ax_anim.set_aspect('equal')
    ax_anim.set_xticks([])
    ax_anim.set_yticks([])
    ax_anim.set_title("Buồng Phân rã", fontsize=14, pad=20)
    for spine in ax_anim.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    scatter = ax_anim.scatter(positions[:, 0], positions[:, 1], 
                              c='red', s=15, edgecolors='black', linewidths=0.5)
    count_text = ax_anim.text(0.5, 1.02, f"Số hạt: {N0}", ha='center', va='bottom',
                           fontsize=12, color='black', transform=ax_anim.transAxes)

    # (Phần code ax_curve giữ nguyên)
    ax_curve.set_title("Đường cong phân rã")
    ax_curve.set_xlabel("Thời gian (s)")
    ax_curve.set_ylabel("Số hạt nhân phóng xạ còn lại")
    ax_curve.grid(True)
    line_sim, = ax_curve.plot([], [], 'r-', label="Mô phỏng")
    line_theory, = ax_curve.plot([], [], 'b--', label="Lý thuyết")
    ax_curve.legend(loc='upper right', fontsize='small', framealpha=0.5)
    ax_curve.set_yscale('log' if scale_var.get() == 'log' else 'linear')

    # (Phần code ax_diff giữ nguyên)
    ax_diff.set_title("Sai khác Tuyệt đối")
    ax_diff.set_xlabel("Thời gian (s)")
    ax_diff.set_ylabel("Mô phỏng - Lý thuyết")
    ax_diff.grid(True)
    line_diff, = ax_diff.plot([], [], 'g-', label="Sai khác tuyệt đối")
    #ax_diff.legend(loc='upper right', fontsize='small', framealpha=0.5)

    # --- THÊM: KHỞI TẠO ĐỒ THỊ MỚI ---
    ax_rel_diff.set_title("Sai khác Tương đối (%)")
    ax_rel_diff.set_xlabel("Thời gian (s)")
    ax_rel_diff.set_ylabel("(Mô phỏng - LT) / LT * 100")
    ax_rel_diff.grid(True)
    line_rel_diff, = ax_rel_diff.plot([], [], 'm-', label="Sai khác tương đối") # 'm' = magenta
    #ax_rel_diff.legend(loc='upper right', fontsize='small', framealpha=0.5)
    # --- KẾT THÚC THÊM ---

    fig.tight_layout() 
    frames = int(total_time / dt)

    # (Cập nhật điểm t=0)
    times.append(0)
    simulated_counts.append(N0)
    theoretical_counts.append(N0)
    diff_log.append(0)
    # --- THÊM ---
    rel_diff_log.append(0) 
    # --- KẾT THÚC THÊM ---

    line_sim.set_data(times, simulated_counts)
    line_theory.set_data(times, theoretical_counts)
    line_diff.set_data(times, diff_log)
    # --- THÊM ---
    line_rel_diff.set_data(times, rel_diff_log)
    # --- KẾT THÚC THÊM ---

    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False, repeat=False)
    canvas.draw()
    status_label.config(text=f"Đang chạy mô phỏng... (Seed {'cố định' if use_fixed_seed else 'ngẫu nhiên'})")
def toggle_decayed_visibility():
    """
    Cập nhật hiển thị của các hạt đã phân rã 
    khi checkbox được_click (đặc biệt khi đang_pause).
    """
    if scatter is None or alive is None:
        return # Chưa bắt đầu simulation

    hide = hide_decayed_var.get()
    
    # Tạo lại mảng màu
    colors = np.zeros((N0, 4))
    colors[alive] = [1, 0, 0, 1.0] # Đỏ
    colors[~alive] = [0.5, 0.5, 0.5, 0.3] # Xám

    if hide:
        # Chỉ hiển thị các hạt còn sống
        scatter.set_offsets(positions[alive])
        scatter.set_facecolor(colors[alive])
    else:
        # Hiển thị tất cả
        scatter.set_offsets(positions)
        scatter.set_facecolor(colors)
    
    # Vẽ lại canvas
    canvas.draw_idle()
    
def toggle_pause():
    global stop_requested, ani
    if ani is None:
        status_label.config(text="Chưa có mô phỏng để dừng/tiếp tục.")
        return
    if stop_requested:
        stop_requested = False
        ani.event_source.start()
        status_label.config(text="Tiếp tục mô phỏng...")
    else:
        stop_requested = True
        ani.event_source.stop()
        status_label.config(text="Đã dừng mô phỏng.")

def toggle_seed():
    global use_fixed_seed
    use_fixed_seed = not use_fixed_seed
    seed_button.config(text=f"Seed: {'Cố định' if use_fixed_seed else 'Ngẫu nhiên'}")

def save_csv():
    # Kiểm tra xem có dữ liệu không
    if len(times) == 0:
        status_label.config(text="Không có dữ liệu để lưu!")
        return
        
    # Thêm 'rel_diff_log' (danh sách sai khác tương đối) vào kiểm tra
    if len(times) != len(rel_diff_log):
        status_label.config(text="Lỗi: Dữ liệu không đồng bộ, không thể lưu.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    
    if file_path:
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # --- SỬA ĐỔI 1: Thêm tiêu đề cột mới ---
                writer.writerow([
                    "Thời gian (s)", 
                    "Mô phỏng (N)", 
                    "Lý thuyết (N)", 
                    "Sai khác tuyệt đối (N)", 
                    "Sai khác tương đối (%)"
                ])
                
                # --- SỬA ĐỔI 2: Thêm 'rel_diff_log' vào zip và ghi hàng ---
                for t, sim, theo, diff_abs, diff_rel in zip(
                    times, 
                    simulated_counts, 
                    theoretical_counts, 
                    diff_log, 
                    rel_diff_log
                ):
                    writer.writerow([t, sim, theo, diff_abs, diff_rel])
                    
            status_label.config(text=f"Đã lưu CSV: {file_path}")
        except Exception as e:
            status_label.config(text=f"Lỗi khi lưu CSV: {e}")

def toggle_scale():
    ax_curve.set_yscale('log' if scale_var.get() == 'log' else 'linear')
    canvas.draw()

# --- Giao diện (ĐÃ ĐƯỢC THIẾT KẾ LẠI) ---
root = tk.Tk()
root.title("Mô phỏng Phân rã Hạt nhân")
root.geometry("1300x900")

style = ttk.Style(root)
style.theme_use('clam') # Sử dụng theme 'clam' cho đẹp

main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# --- Khung chứa tất cả các điều khiển (ở trên) ---
controls_container = ttk.Frame(main_frame)
controls_container.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

# --- Nhóm 1: Thông số Mô phỏng ---
frame_inputs = ttk.LabelFrame(controls_container, text="Thông số Mô phỏng", padding="10")
frame_inputs.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), pady=5)

# (Phần này giữ nguyên code cũ của bạn)
ttk.Label(frame_inputs, text="Số hạt ban đầu (N0):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
entry_particles = ttk.Entry(frame_inputs, width=12)
entry_particles.grid(row=0, column=1, padx=5, pady=5)
entry_particles.insert(0, "500")

ttk.Label(frame_inputs, text="Chu kỳ bán rã (s):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
entry_half_life = ttk.Entry(frame_inputs, width=12)
entry_half_life.grid(row=1, column=1, padx=5, pady=5)
entry_half_life.insert(0, "30")

ttk.Label(frame_inputs, text="Thời gian mô phỏng (s):").grid(row=0, column=2, sticky="e", padx=5, pady=5)
entry_total_time = ttk.Entry(frame_inputs, width=12)
entry_total_time.grid(row=0, column=3, padx=5, pady=5)
entry_total_time.insert(0, "200")

ttk.Label(frame_inputs, text="Bước thời gian (s):").grid(row=1, column=2, sticky="e", padx=5, pady=5)
entry_time_step = ttk.Entry(frame_inputs, width=12)
entry_time_step.grid(row=1, column=3, padx=5, pady=5)
entry_time_step.insert(0, "1")


# --- Nhóm 2: Nút điều khiển ---
button_frame = ttk.LabelFrame(controls_container, text="Điều khiển", padding="10")
button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

# (Phần này giữ nguyên code cũ của bạn)
ttk.Button(button_frame, text="Bắt đầu", command=start_simulation).grid(row=0, column=0, padx=5, pady=2)
ttk.Button(button_frame, text="Dừng / Tiếp tục", command=toggle_pause).grid(row=0, column=1, padx=5, pady=2)
seed_button = ttk.Button(button_frame, text="Seed: Cố định", command=toggle_seed)
seed_button.grid(row=1, column=0, padx=5, pady=2)
ttk.Button(button_frame, text="Lưu CSV", command=save_csv).grid(row=1, column=1, padx=5, pady=2)


# --- Nhóm 3: Tùy chọn Hiển thị (ĐÃ CẬP NHẬT) ---
scale_frame = ttk.LabelFrame(controls_container, text="Tùy chọn Hiển thị", padding="10")
scale_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

# (Phần Radiobutton giữ nguyên)
scale_var = tk.StringVar(value='linear')
ttk.Radiobutton(scale_frame, text="Thang Tuyến tính", variable=scale_var, value='linear', command=toggle_scale).pack(anchor=tk.W, padx=5, pady=2)
ttk.Radiobutton(scale_frame, text="Thang Logarit", variable=scale_var, value='log', command=toggle_scale).pack(anchor=tk.W, padx=5, pady=2)

# --- THÊM MỚI ---
hide_decayed_var = tk.BooleanVar(value=False) # Khởi tạo biến
chk_hide = ttk.Checkbutton(scale_frame, 
                           text="Ẩn hạt đã phân rã", 
                           variable=hide_decayed_var, 
                           onvalue=True, 
                           offvalue=False,
                           command=toggle_decayed_visibility) # Gọi hàm mới
chk_hide.pack(anchor=tk.W, padx=5, pady=2)
# --- KẾT THÚC THÊM MỚI ---


# --- Đồ thị (Canvas) ---
fig = plt.Figure(figsize=(12, 9))
canvas = FigureCanvasTkAgg(fig, master=main_frame)
# LƯU Ý: CHƯA PACK CANVAS VỘI


# --- SỬA LỖI HIỂN THỊ: PACK BOTTOM TRƯỚC KHI PACK CANVAS ---

# 1. Thanh Trạng thái (Pack TRƯỚC để nằm DƯỚI CÙNG)
status_label = ttk.Label(main_frame, text="Sẵn sàng.", relief=tk.SUNKEN, anchor=tk.W, padding=5)
status_label.pack(side=tk.BOTTOM, fill=tk.X)

# 2. Ghi chú tác giả (Pack SAU để nằm NGAY TRÊN thanh trạng thái)
credit_label = ttk.Label(main_frame, 
                         text="Phần mềm được viết bởi PGS. TS. Nguyễn Ngọc Anh, Đại học Phenikaa",
                         font=("Arial", 9, "italic"),
                         anchor=tk.E, 
                         padding=(0, 0, 10, 5))
credit_label.pack(side=tk.BOTTOM, fill=tk.X)

# 3. Pack Canvas CUỐI CÙNG
# Nó sẽ tự động lấp đầy không gian còn lại ở giữa (trên TOP)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# --- KẾT THÚC SỬA ---


root.mainloop()