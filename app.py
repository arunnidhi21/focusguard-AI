"""
FocusGuard AI — GUI
====================
Loads the model trained by train.py (focusguard_model.pkl) and runs
the interactive Tkinter dashboard. Run `python train.py` first if
the pickle doesn't exist yet.
"""

import math
import os
import sys
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import joblib
import pandas as pd

MODEL_PATH = "focusguard_model.pkl"

if not os.path.exists(MODEL_PATH):
    print(
        f"'{MODEL_PATH}' not found. Run `python train.py` first "
        "to train and save the model."
    )
    sys.exit(1)

artifact = joblib.load(MODEL_PATH)
preprocessor = artifact["preprocessor"]
model = artifact["model"]
FEATURE_COLUMNS = artifact["feature_columns"]
feature_importance = artifact["feature_importance"]
CV_ACCURACY = artifact["cv_accuracy"]
BASELINE = artifact["majority_baseline"]
N_SAMPLES = artifact["n_samples"]

# ============================================================
# THEME
# ============================================================
BG = "#050b15"
PANEL = "#0c1728"
CARD = "#111f35"
CARD2 = "#172a46"
SHADOW = "#02050a"
BORDER = "#244262"
CYAN = "#00d9ff"
CYAN_DARK = "#087f9b"
GREEN = "#00e676"
RED = "#ff4d5a"
YELLOW = "#ffd54f"
WHITE = "#f4fbff"
MUTED = "#9db2cc"
TRACK = "#263a59"

root = tk.Tk()
root.title("FocusGuard AI")
root.geometry("980x920")
root.configure(bg=BG)
root.resizable(False, False)

# ============================================================
# VARIABLES
# ============================================================
previous_app_var = tk.StringVar(value="WhatsApp")
notification_var = tk.StringVar(value="yes")
opens_30_var = tk.StringVar(value="1")
opens_2hr_var = tk.StringVar(value="2")
previous_duration_var = tk.StringVar(value="30")
time_since_var = tk.StringVar(value="30")
hour_var = tk.StringVar(value="14")
analysis_running = False


# ============================================================
# 3D HELPERS
# ============================================================
def make_card(parent, height=None, padx=25, pady=8):
    shadow = tk.Frame(parent, bg=SHADOW)
    shadow.pack(fill="x", padx=padx, pady=pady)
    if height:
        shadow.configure(height=height)
        shadow.pack_propagate(False)

    body = tk.Frame(
        shadow, bg=CARD,
        highlightbackground=BORDER,
        highlightthickness=1
    )
    body.pack(fill="both", expand=True, padx=(0, 6), pady=(0, 6))
    tk.Frame(body, bg=CARD2, height=3).pack(fill="x")
    content = tk.Frame(body, bg=CARD)
    content.pack(fill="both", expand=True)
    return shadow, content


def make_3d_button(parent, text, command, width, primary=False):
    normal = CYAN if primary else CARD2
    hover = "#51e7ff" if primary else "#29496f"
    fg = BG if primary else WHITE

    holder = tk.Frame(parent, bg=SHADOW)
    button = tk.Button(
        holder, text=text, command=command, width=width,
        font=("Arial", 11, "bold"), bg=normal, fg=fg,
        activebackground=hover, activeforeground=BG,
        relief="flat", bd=0, cursor="hand2", padx=8, pady=10
    )
    button.pack(padx=(0, 5), pady=(0, 5))

    def enter(_):
        button.configure(bg=hover)

    def leave(_):
        button.configure(bg=normal)
        button.place_configure(x=0, y=0)

    def press(_):
        button.place_configure(x=2, y=2)

    def release(_):
        button.place_configure(x=0, y=0)

    button.bind("<Enter>", enter)
    button.bind("<Leave>", leave)
    button.bind("<ButtonPress-1>", press)
    button.bind("<ButtonRelease-1>", release)
    return holder, button


def animate_title(step=0):
    colors = [WHITE, "#d8fbff", "#7feaff", WHITE]
    title.configure(fg=colors[step % len(colors)])
    root.after(420, lambda: animate_title(step + 1))


def animate_core(step=0):
    if not root.winfo_exists():
        return
    pulse = 3 + int(3 * math.sin(step * 0.16))
    core_canvas.coords(core_outer, 64-pulse, 64-pulse, 192+pulse, 192+pulse)
    core_canvas.itemconfigure(core_outer, width=5 + pulse // 2)
    root.after(60, lambda: animate_core(step + 1))


def animate_status(step=0):
    if not analysis_running:
        return
    dots = "." * (step % 4)
    status_label.configure(text=f"AI is analyzing session{dots}")
    root.after(180, lambda: animate_status(step + 1))


def animate_bar(canvas, item, target, current=0.0):
    target = max(0.0, min(100.0, float(target)))
    current = min(current, target)
    canvas.coords(item, 0, 0, current * 4.65, 20)
    if current < target:
        root.after(8, lambda: animate_bar(canvas, item, target, current + 1.25))


def animate_gauge(target, current=0.0):
    target = max(0.0, min(100.0, float(target)))
    current = min(current, target)
    gauge.itemconfigure(gauge_arc, extent=current * 2.7)
    gauge.itemconfigure(gauge_value, text=f"{current:.0f}%")
    if current < target:
        root.after(10, lambda: animate_gauge(target, current + 1.1))


def set_risk(habitual):
    if habitual >= 60:
        risk_label.configure(text="HIGH DISTRACTION RISK", fg=RED)
        recommendation_label.configure(
            text="Take a short break before opening Instagram.", fg=RED
        )
        risk_dot.configure(fg=RED)
    elif habitual >= 40:
        risk_label.configure(text="MODERATE DISTRACTION RISK", fg=YELLOW)
        recommendation_label.configure(
            text="Be mindful of your Instagram usage.", fg=YELLOW
        )
        risk_dot.configure(fg=YELLOW)
    else:
        risk_label.configure(text="LOW DISTRACTION RISK", fg=GREEN)
        recommendation_label.configure(
            text="This session appears to be intentional.", fg=GREEN
        )
        risk_dot.configure(fg=GREEN)


def reset_prediction_widgets():
    result_label.configure(text="READY", fg=CYAN)
    result_icon.configure(text="◈", fg=CYAN)
    status_label.configure(text="Enter session information and analyze.")
    intentional_value.configure(text="0.00%")
    habitual_value.configure(text="0.00%")
    gauge.itemconfigure(gauge_arc, extent=0, outline=CYAN)
    gauge.itemconfigure(gauge_value, text="0%", fill=GREEN)
    intentional_canvas.coords(intentional_bar, 0, 0, 0, 20)
    habitual_canvas.coords(habitual_bar, 0, 0, 0, 20)
    risk_label.configure(text="", fg=GREEN)
    recommendation_label.configure(text="", fg=MUTED)
    confidence_text.configure(text="MODEL CONFIDENCE\nAwaiting analysis...")
    system_status.configure(text="●  AI MODEL READY", fg=GREEN)
    risk_dot.configure(fg=TRACK)
    for label in signal_value_labels:
        label.configure(text="—", fg=MUTED)


def reset_fields():
    previous_app_var.set("WhatsApp")
    notification_var.set("yes")
    opens_30_var.set("1")
    opens_2hr_var.set("2")
    previous_duration_var.set("30")
    time_since_var.set("30")
    hour_var.set("14")
    reset_prediction_widgets()


# ============================================================
# PREDICTION
# ============================================================
def predict_session():
    global analysis_running
    if analysis_running:
        return

    try:
        analysis_running = True
        analyze_button.configure(text="⏳  ANALYZING...", state="disabled")
        system_status.configure(text="●  MODEL PROCESSING", fg=YELLOW)
        status_label.configure(text="AI is analyzing session...")
        animate_status()

        previous_app = previous_app_var.get().strip()
        notification = notification_var.get().strip().lower()
        opens_30 = int(opens_30_var.get())
        opens_2hr = int(opens_2hr_var.get())
        previous_duration = int(previous_duration_var.get())
        time_since = int(time_since_var.get())
        current_hour = int(hour_var.get())

        if not previous_app:
            raise ValueError("Please select the previous app.")
        if notification not in ("yes", "no"):
            raise ValueError("Notification must be yes or no.")
        if not 0 <= current_hour <= 23:
            raise ValueError("Current hour must be between 0 and 23.")
        if min(opens_30, opens_2hr, previous_duration, time_since) < 0:
            raise ValueError("Values cannot be negative.")

        rapid_reopen = int(time_since <= 15)
        now = datetime.now()
        day_of_week = now.weekday()
        is_weekend = int(day_of_week >= 5)

        row = {
            "hour": current_hour,
            "previous_app": previous_app,
            "instagram_notification": 1 if notification == "yes" else 0,
            "instagram_opens_30min": opens_30,
            "instagram_opens_2hr": opens_2hr,
            "previous_instagram_duration": previous_duration,
            "time_since_previous_instagram": time_since,
            "rapid_reopen": rapid_reopen,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
        }
        new_session = pd.DataFrame([{col: row[col] for col in FEATURE_COLUMNS}])

        encoded = preprocessor.transform(new_session)
        prediction = int(model.predict(encoded)[0])
        probabilities = model.predict_proba(encoded)[0]
        class_probabilities = dict(zip(model.classes_, probabilities))

        habitual = float(class_probabilities.get(0, 0) * 100)
        intentional = float(class_probabilities.get(1, 0) * 100)

        if prediction == 1:
            result_label.configure(text="INTENTIONAL USAGE", fg=GREEN)
            result_icon.configure(text="✓", fg=GREEN)
            status_label.configure(text="This session appears intentional.")
            gauge.itemconfigure(gauge_arc, outline=CYAN)
        else:
            result_label.configure(text="HABITUAL USAGE", fg=RED)
            result_icon.configure(text="!", fg=RED)
            status_label.configure(text="This session appears habitual.")
            gauge.itemconfigure(gauge_arc, outline=RED)

        intentional_value.configure(text=f"{intentional:.2f}%")
        habitual_value.configure(text=f"{habitual:.2f}%")

        intentional_canvas.coords(intentional_bar, 0, 0, 0, 20)
        habitual_canvas.coords(habitual_bar, 0, 0, 0, 20)
        gauge.itemconfigure(gauge_arc, extent=0)
        gauge.itemconfigure(gauge_value, text="0%")

        animate_bar(intentional_canvas, intentional_bar, intentional)
        animate_bar(habitual_canvas, habitual_bar, habitual)
        animate_gauge(max(intentional, habitual))

        set_risk(habitual)

        confidence = max(intentional, habitual)
        confidence_text.configure(
            text=(
                "MODEL CONFIDENCE\n"
                f"{confidence:.1f}% confidence • "
                f"Prediction class: {prediction}"
            )
        )

        top_features = feature_importance.head(4)
        for idx, (_, frow) in enumerate(top_features.iterrows()):
            name = str(frow["feature"]).replace("num__", "").replace(
                "cat__", ""
            ).replace("remainder__", "")
            signal_value_labels[idx].configure(
                text=f"{name}   {float(frow['importance']):.3f}",
                fg=CYAN if idx == 0 else WHITE
            )

        system_status.configure(text="●  AI MODEL READY", fg=GREEN)

    except ValueError as exc:
        messagebox.showerror("Input Error", str(exc))
        status_label.configure(text="Please correct the input and try again.")
    except Exception as exc:
        messagebox.showerror("Prediction Error", f"{type(exc).__name__}: {exc}")
        print("Prediction Error:", repr(exc))
        status_label.configure(text="Prediction failed. Check the console.")
    finally:
        analysis_running = False
        analyze_button.configure(text="⚡  ANALYZE USAGE", state="normal")


# ============================================================
# HEADER
# ============================================================
header = tk.Frame(root, bg=BG)
header.pack(pady=(16, 7))

title = tk.Label(
    header, text="FOCUSGUARD AI",
    font=("Arial", 30, "bold"), fg=WHITE, bg=BG
)
title.pack()

tk.Label(
    header, text="INSTAGRAM USAGE INTENT PREDICTOR",
    font=("Arial", 10, "bold"), fg=CYAN, bg=BG
).pack(pady=3)

system_status = tk.Label(
    header, text="●  AI MODEL READY",
    font=("Arial", 8, "bold"), fg=GREEN, bg=BG
)
system_status.pack()

tk.Label(
    header,
    text=(
        f"Model: RandomForest • {N_SAMPLES} training sessions • "
        f"CV accuracy {CV_ACCURACY*100:.1f}% "
        f"(baseline {BASELINE*100:.1f}%)"
    ),
    font=("Arial", 8), fg=MUTED, bg=BG
).pack(pady=(2, 0))

# ============================================================
# INPUT CARD
# ============================================================
_, input_card = make_card(root, height=275, padx=25, pady=6)

tk.Label(
    input_card, text="SESSION ANALYSIS",
    font=("Arial", 15, "bold"), fg=CYAN, bg=CARD
).grid(row=0, column=0, columnspan=2, pady=(8, 8))

label_style = {"font": ("Arial", 9, "bold"), "fg": MUTED, "bg": CARD}

fields = [
    ("Previous App", 1),
    ("Instagram Notification", 2),
    ("Instagram Opens (30 min)", 3),
    ("Instagram Opens (2 hr)", 4),
    ("Previous Instagram Duration (min)", 5),
    ("Time Since Previous Instagram (min)", 6),
    ("Current Hour (0-23)", 7),
]

for text, row in fields:
    tk.Label(input_card, text=text, **label_style).grid(
        row=row, column=0, sticky="w", padx=(30, 12), pady=3
    )

previous_app_box = ttk.Combobox(
    input_card, textvariable=previous_app_var,
    values=["Facebook", "Spotify", "Messages", "Camera", "Chrome", "WhatsApp", "Settings", "YouTube"],
    state="readonly", width=25
)
previous_app_box.grid(row=1, column=1, padx=(5, 30), pady=3)

notification_box = ttk.Combobox(
    input_card, textvariable=notification_var,
    values=["yes", "no"], state="readonly", width=25
)
notification_box.grid(row=2, column=1, padx=(5, 30), pady=3)


def add_entry(variable, row):
    e = tk.Entry(
        input_card, textvariable=variable, width=27,
        font=("Arial", 9), bg="#f7fbff", fg="#111111",
        relief="flat", bd=0, insertbackground="#111111"
    )
    e.grid(row=row, column=1, padx=(5, 30), pady=3, ipady=3)
    return e

add_entry(opens_30_var, 3)
add_entry(opens_2hr_var, 4)
add_entry(previous_duration_var, 5)
add_entry(time_since_var, 6)
add_entry(hour_var, 7)

# ============================================================
# BUTTONS
# ============================================================
button_frame = tk.Frame(root, bg=BG)
button_frame.pack(pady=5)

_, analyze_button = make_3d_button(
    button_frame, "⚡  ANALYZE USAGE", predict_session, 20, True
)
analyze_button.master.grid(row=0, column=0, padx=6)

_, reset_button = make_3d_button(
    button_frame, "↻  RESET", reset_fields, 11, False
)
reset_button.master.grid(row=0, column=1, padx=6)

# ============================================================
# RESULT CARD
# ============================================================
_, result_card = make_card(root, height=500, padx=25, pady=6)

result_icon = tk.Label(
    result_card, text="◈", font=("Arial", 22, "bold"),
    fg=CYAN, bg=CARD
)
result_icon.pack(pady=(5, 0))

result_label = tk.Label(
    result_card, text="READY", font=("Arial", 22, "bold"),
    fg=CYAN, bg=CARD
)
result_label.pack(pady=(0, 2))

status_label = tk.Label(
    result_card, text="Enter session information and analyze.",
    font=("Arial", 9), fg=MUTED, bg=CARD
)
status_label.pack(pady=(0, 5))

# ============================================================
# CENTER: AI CORE + FEATURE SIGNALS
# ============================================================
center = tk.Frame(result_card, bg=CARD)
center.pack(fill="x", padx=35)

core_frame = tk.Frame(center, bg=CARD)
core_frame.pack(side="left", padx=(15, 30))

core_canvas = tk.Canvas(
    core_frame, width=260, height=210,
    bg=CARD, highlightthickness=0
)
core_canvas.pack()

core_canvas.create_oval(55, 55, 205, 205, outline=TRACK, width=18)
core_outer = core_canvas.create_oval(
    64, 64, 196, 196, outline=CYAN, width=5
)
core_canvas.create_oval(76, 76, 184, 184, outline="#355275", width=2)
core_canvas.create_oval(89, 89, 171, 171, fill="#0b2031", outline=CYAN_DARK, width=2)
core_canvas.create_text(
    130, 111, text="AI", fill=WHITE,
    font=("Arial", 24, "bold")
)
core_canvas.create_text(
    130, 140, text="CORE", fill=CYAN,
    font=("Arial", 8, "bold")
)

gauge = tk.Canvas(
    core_frame, width=220, height=220,
    bg=CARD, highlightthickness=0
)
gauge.place(x=20, y=-5)

gauge.create_arc(
    22, 22, 198, 198,
    start=225, extent=270,
    style="arc", outline=TRACK, width=12
)
gauge_arc = gauge.create_arc(
    22, 22, 198, 198,
    start=225, extent=0,
    style="arc", outline=CYAN, width=12
)
gauge.create_text(110, 83, text="CONFIDENCE", fill=MUTED, font=("Arial", 8, "bold"))
gauge_value = gauge.create_text(110, 111, text="0%", fill=GREEN, font=("Arial", 25, "bold"))
gauge.create_text(110, 139, text="AI PREDICTION", fill=MUTED, font=("Arial", 8, "bold"))

signals = tk.Frame(
    center, bg="#0a1423",
    highlightbackground=BORDER, highlightthickness=1
)
signals.pack(side="right", fill="both", expand=True, pady=8)

tk.Label(
    signals, text="MODEL SIGNALS",
    font=("Arial", 12, "bold"), fg=CYAN, bg="#0a1423"
).pack(pady=(10, 2))

tk.Label(
    signals,
    text="Global Random Forest importance",
    font=("Arial", 8), fg=MUTED, bg="#0a1423"
).pack(pady=(0, 8))

signal_value_labels = []
for _ in range(4):
    lbl = tk.Label(
        signals, text="—", anchor="w",
        font=("Consolas", 8, "bold"),
        fg=MUTED, bg="#0a1423"
    )
    lbl.pack(fill="x", padx=12, pady=5)
    signal_value_labels.append(lbl)

# ============================================================
# PROBABILITY BARS
# ============================================================
def probability_row(parent, title_text, color):
    frame = tk.Frame(parent, bg=CARD)
    frame.pack(fill="x", padx=45, pady=3)

    top = tk.Frame(frame, bg=CARD)
    top.pack(fill="x")

    tk.Label(
        top, text=title_text, font=("Arial", 8, "bold"),
        fg=MUTED, bg=CARD
    ).pack(side="left")

    value = tk.Label(
        top, text="0.00%", font=("Arial", 10, "bold"),
        fg=color, bg=CARD
    )
    value.pack(side="right")

    canvas = tk.Canvas(
        frame, width=465, height=20,
        bg=TRACK, highlightthickness=0
    )
    canvas.pack(pady=(2, 0))
    bar = canvas.create_rectangle(0, 0, 0, 20, fill=color, outline="")
    return value, canvas, bar

intentional_value, intentional_canvas, intentional_bar = probability_row(
    result_card, "INTENTIONAL USAGE", GREEN
)
habitual_value, habitual_canvas, habitual_bar = probability_row(
    result_card, "HABITUAL USAGE", RED
)

# ============================================================
# RISK PANEL
# ============================================================
risk_panel = tk.Frame(
    result_card, bg="#091421",
    highlightbackground=BORDER, highlightthickness=1
)
risk_panel.pack(fill="x", padx=45, pady=(5, 3))

risk_dot = tk.Label(
    risk_panel, text="●", font=("Arial", 13, "bold"),
    fg=TRACK, bg="#091421"
)
risk_dot.pack(side="left", padx=(14, 5))

risk_label = tk.Label(
    risk_panel, text="", font=("Arial", 12, "bold"),
    fg=GREEN, bg="#091421"
)
risk_label.pack(side="left", pady=7)

recommendation_label = tk.Label(
    risk_panel, text="", font=("Arial", 8),
    fg=MUTED, bg="#091421"
)
recommendation_label.pack(side="right", padx=14)

confidence_text = tk.Label(
    result_card, text="MODEL CONFIDENCE\nAwaiting analysis...",
    font=("Arial", 8, "bold"), fg=MUTED, bg=CARD
)
confidence_text.pack(pady=2)

# ============================================================
# FOOTER
# ============================================================
tk.Label(
    root,
    text="FOCUSGUARD AI   •   RANDOM FOREST   •   MACHINE LEARNING",
    font=("Arial", 8, "bold"), fg=MUTED, bg=BG
).pack(side="bottom", pady=4)

# ============================================================
# START
# ============================================================
reset_fields()
analyze_button.focus_set()
animate_title()
animate_core()
root.mainloop()
