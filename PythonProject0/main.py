"""
Module 5: GUI Frontend integrating M1-M4
- Tkinter-based interface
- Sidebar to select module functions
- Dynamic input fields per function
- Auto-opens first generated image
- Exit button
"""

import os
import sys
import glob
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk   # pip install pillow

# Import the four modules
from data_pipeline      import run_pipeline, quality_report, clean_data, build_features
from picture_generation import (analyze_time_pattern, analyze_zone_hotspot,
                                analyze_fare_factors, analyze_speed_period)
from data_prediction    import build_demand_table
from AI_assistance      import llm_parse, HANDLERS

OUT_DIR    = "outputs"
DATA_PATH  = "yellow_tripdata_2026-01.parquet"
os.makedirs(OUT_DIR, exist_ok=True)


# ============ Helpers ============
def open_file_external(path: str):
    """Open a file with the OS default viewer."""
    if not os.path.exists(path):
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"[open error] {e}")


def list_new_images(before: set):
    """Return new png files in OUT_DIR sorted by mtime ascending."""
    after  = set(glob.glob(os.path.join(OUT_DIR, "*.png")))
    new    = list(after - before)
    new.sort(key=lambda p: os.path.getmtime(p))
    return new


# ============ Main GUI ============
class TaxiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NYC Taxi Analytics — GUI")
        self.geometry("1100x680")
        self.configure(bg="#f4f6fa")

        self.df = None                    # cached dataframe
        self._build_layout()
        self._bind_modules()
        self._show_welcome()
        # Load data in background to keep UI responsive
        self.after(200, self._async_load_data)

    # ---------- layout ----------
    def _build_layout(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", padding=6, font=("Segoe UI", 10))
        style.configure("Side.TButton", padding=8, font=("Segoe UI", 10, "bold"))
        style.configure("Quit.TButton", padding=8, font=("Segoe UI", 10, "bold"),
                        foreground="white", background="#c0392b")

        # Sidebar
        sidebar = tk.Frame(self, bg="#2c3e50", width=230)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="🚖 Taxi Analytics",
                 bg="#2c3e50", fg="white",
                 font=("Segoe UI", 14, "bold")).pack(pady=18)

        self.module_buttons = []
        modules = [
            ("M1  Data Pipeline",      self.show_m1),
            ("M2  Visualizations",     self.show_m2),
            ("M3  Demand Prediction",  self.show_m3),
            ("M4  AI Assistant (Q&A)", self.show_m4),
        ]
        for text, cmd in modules:
            b = tk.Button(sidebar, text=text, command=cmd,
                          bg="#34495e", fg="white", relief="flat",
                          font=("Segoe UI", 10, "bold"),
                          activebackground="#1abc9c",
                          activeforeground="white",
                          height=2, anchor="w", padx=15)
            b.pack(fill="x", padx=10, pady=4)
            self.module_buttons.append(b)

        tk.Frame(sidebar, bg="#2c3e50").pack(expand=True, fill="both")

        tk.Button(sidebar, text="✕  Exit", command=self.on_exit,
                  bg="#c0392b", fg="white", relief="flat",
                  font=("Segoe UI", 11, "bold"),
                  activebackground="#e74c3c", activeforeground="white",
                  height=2).pack(side="bottom", fill="x", padx=10, pady=14)

        # Right area: header + form + output
        self.main = tk.Frame(self, bg="#f4f6fa")
        self.main.pack(side="right", fill="both", expand=True)

        self.header = tk.Label(self.main, text="", bg="#f4f6fa",
                               font=("Segoe UI", 16, "bold"),
                               fg="#2c3e50", anchor="w")
        self.header.pack(fill="x", padx=20, pady=(15, 5))

        self.desc = tk.Label(self.main, text="", bg="#f4f6fa",
                             font=("Segoe UI", 10), fg="#555",
                             anchor="w", justify="left", wraplength=820)
        self.desc.pack(fill="x", padx=20)

        self.form = tk.Frame(self.main, bg="#f4f6fa")
        self.form.pack(fill="x", padx=20, pady=10)

        # Output split: log (left) + image preview (right)
        body = tk.Frame(self.main, bg="#f4f6fa")
        body.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        log_frame = tk.LabelFrame(body, text=" Output ", bg="#f4f6fa",
                                  font=("Segoe UI", 10, "bold"), fg="#2c3e50")
        log_frame.pack(side="left", fill="both", expand=True)

        self.log = scrolledtext.ScrolledText(log_frame, wrap="word",
                                             font=("Consolas", 10),
                                             bg="white", height=15)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)

        prev_frame = tk.LabelFrame(body, text=" Preview ", bg="#f4f6fa",
                                   font=("Segoe UI", 10, "bold"),
                                   fg="#2c3e50", width=380)
        prev_frame.pack(side="right", fill="both", padx=(10, 0))
        prev_frame.pack_propagate(False)

        self.preview = tk.Label(prev_frame, bg="white",
                                text="(image preview area)",
                                fg="#999", font=("Segoe UI", 9))
        self.preview.pack(fill="both", expand=True, padx=5, pady=5)
        self._preview_imgtk = None

        # Status bar
        self.status = tk.Label(self, text="Loading data...",
                               bd=1, relief="sunken", anchor="w",
                               bg="#ecf0f1", font=("Segoe UI", 9))
        self.status.pack(side="bottom", fill="x")

    # ---------- module bindings ----------
    def _bind_modules(self):
        self.module_actions = {
            "M1": self._render_m1,
            "M2": self._render_m2,
            "M3": self._render_m3,
            "M4": self._render_m4,
        }

    def _show_welcome(self):
        self.header.config(text="Welcome")
        self.desc.config(text=(
            "Use the left sidebar to choose a module:\n"
            "  • M1 inspect data quality   • M2 generate analytic charts\n"
            "  • M3 train demand models    • M4 ask natural-language questions"
        ))

    def _clear_form(self):
        for w in self.form.winfo_children():
            w.destroy()

    def _log(self, msg=""):
        self.log.insert("end", str(msg) + "\n")
        self.log.see("end")
        self.update_idletasks()

    def _set_status(self, txt):
        self.status.config(text=txt); self.update_idletasks()

    # ---------- async data load ----------
    def _async_load_data(self):
        def task():
            try:
                self._log(">>> Loading & cleaning data via M1.run_pipeline()...")
                self.df = run_pipeline(DATA_PATH)
                self._log(f">>> Ready. Rows = {len(self.df)}\n")
                self._set_status(f"Data loaded · rows={len(self.df)}")
            except Exception as e:
                self._log(f"[load error] {e}")
                self._set_status("Data load failed")
                messagebox.showerror("Load Error", str(e))
        threading.Thread(target=task, daemon=True).start()

    def _ensure_data(self):
        if self.df is None:
            messagebox.showinfo("Please wait", "Data is still loading.")
            return False
        return True

    # ---------- preview ----------
    def show_image(self, path: str):
        if not path or not os.path.exists(path):
            return
        try:
            img = Image.open(path)
            img.thumbnail((360, 480))
            self._preview_imgtk = ImageTk.PhotoImage(img)
            self.preview.config(image=self._preview_imgtk, text="")
        except Exception as e:
            self._log(f"[preview error] {e}")

    def run_with_charts(self, fn, *args, **kwargs):
        """Run a function that produces images, auto-open the first new image."""
        before = set(glob.glob(os.path.join(OUT_DIR, "*.png")))
        result = fn(*args, **kwargs)
        new = list_new_images(before)
        if new:
            first = new[0]
            self._log(f"Generated {len(new)} chart(s). First: {first}")
            for p in new:
                self._log(f"  - {p}")
            self.show_image(first)
            open_file_external(first)
        return result

    # ============ M1 ============
    def show_m1(self):
        self.header.config(text="M1 · Data Pipeline")
        self.desc.config(text="Inspect data quality, run cleaning, view feature stats.")
        self._clear_form()

        ttk.Button(self.form, text="Show Quality Report",
                   command=self._m1_report).pack(side="left", padx=4)
        ttk.Button(self.form, text="Show Cleaned Stats",
                   command=self._m1_stats).pack(side="left", padx=4)
        ttk.Button(self.form, text="Reload Data",
                   command=self._async_load_data).pack(side="left", padx=4)

    def _m1_report(self):
        if not self._ensure_data(): return
        self._log("\n--- Re-running quality_report on raw-style snapshot ---")
        try:
            quality_report(self.df)   # prints internally
            self._log("(see console for full dict)")
        except Exception as e:
            self._log(f"[error] {e}")

    def _m1_stats(self):
        if not self._ensure_data(): return
        self._log("\n--- Cleaned dataframe summary ---")
        self._log(f"shape: {self.df.shape}")
        self._log(f"columns: {list(self.df.columns)}")
        self._log(self.df[["trip_distance", "fare_amount",
                           "duration_min", "avg_speed_mph"]]
                       .describe().round(2).to_string())

    # ============ M2 ============
    def show_m2(self):
        self.header.config(text="M2 · Visualization")
        self.desc.config(text="Generate analytic charts. The first chart opens automatically.")
        self._clear_form()

        opts = [
            ("Time Pattern",  analyze_time_pattern),
            ("Zone Hotspot",  analyze_zone_hotspot),
            ("Fare Factors",  analyze_fare_factors),
            ("Speed vs Time", analyze_speed_period),
            ("ALL of above",  None),
        ]
        for txt, fn in opts:
            ttk.Button(self.form, text=txt,
                       command=lambda f=fn, t=txt: self._m2_run(f, t)
                       ).pack(side="left", padx=3)

    def _m2_run(self, fn, label):
        if not self._ensure_data(): return
        self._log(f"\n>>> M2 {label}")
        def task():
            try:
                if fn is None:
                    self.run_with_charts(self._m2_all)
                else:
                    self.run_with_charts(fn, self.df)
                self._set_status(f"M2 {label} done")
            except Exception as e:
                self._log(f"[error] {e}")
        threading.Thread(target=task, daemon=True).start()

    def _m2_all(self):
        analyze_time_pattern(self.df)
        analyze_zone_hotspot(self.df)
        analyze_fare_factors(self.df)
        analyze_speed_period(self.df)

    # ============ M3 ============
    def show_m3(self):
        self.header.config(text="M3 · Demand Prediction")
        self.desc.config(text="Train MLP & Random Forest, compare MAE/RMSE.")
        self._clear_form()

        ttk.Label(self.form, text="Top-K zones:",
                  background="#f4f6fa").pack(side="left", padx=4)
        self.m3_topk = tk.StringVar(value="30")
        ttk.Entry(self.form, textvariable=self.m3_topk, width=6
                  ).pack(side="left", padx=2)

        ttk.Label(self.form, text="MLP epochs:",
                  background="#f4f6fa").pack(side="left", padx=4)
        self.m3_epochs = tk.StringVar(value="20")
        ttk.Entry(self.form, textvariable=self.m3_epochs, width=6
                  ).pack(side="left", padx=2)

        ttk.Button(self.form, text="Run Comparison",
                   command=self._m3_run).pack(side="left", padx=8)

    def _m3_run(self):
        if not self._ensure_data(): return
        try:
            topk   = int(self.m3_topk.get())
            epochs = int(self.m3_epochs.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Top-K / epochs must be int")
            return
        self._log(f"\n>>> M3 Training (topK={topk}, epochs={epochs}) ...")
        self._set_status("Training models, please wait...")

        def task():
            try:
                self.run_with_charts(self._m3_pipeline, topk, epochs)
                self._set_status("M3 done")
            except Exception as e:
                self._log(f"[error] {e}")
                self._set_status("M3 failed")
        threading.Thread(target=task, daemon=True).start()

    def _m3_pipeline(self, topk, epochs):
        """Inline mini-version of M3 main, so we can pass GUI params."""
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing   import StandardScaler
        from sklearn.ensemble        import RandomForestRegressor
        from sklearn.metrics         import mean_absolute_error, mean_squared_error
        from data_prediction import train_mlp, plot_loss, plot_comparison, plot_metrics_bar

        demand = build_demand_table(self.df, top_k_zones=topk)
        feat_cols = ["PULocationID", "pickup_hour", "weekday",
                     "is_weekend", "is_rush_hour",
                     "hour_sin", "hour_cos", "wday_sin", "wday_cos"]
        X = demand[feat_cols].values
        y = demand["demand"].values.astype(np.float32)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

        self._log("Training MLP...")
        mlp_pred, hist = train_mlp(X_tr_s, y_tr, X_te_s, y_te, epochs=epochs)
        plot_loss(hist)

        self._log("Training RandomForest...")
        rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
        rf.fit(X_tr, y_tr); rf_pred = rf.predict(X_te)

        def m(name, pred):
            mae  = mean_absolute_error(y_te, pred)
            rmse = np.sqrt(mean_squared_error(y_te, pred))
            self._log(f"  {name:>13s}  MAE={mae:.3f}  RMSE={rmse:.3f}")
            return {"MAE": mae, "RMSE": rmse}

        self._log("=== Test set ===")
        metrics = {"MLP": m("MLP", mlp_pred),
                   "RandomForest": m("RandomForest", rf_pred)}
        plot_comparison(y_te, mlp_pred, rf_pred)
        plot_metrics_bar(metrics)

    # ============ M4 ============
    def show_m4(self):
        self.header.config(text="M4 · AI Assistant (Q&A)")
        self.desc.config(text="Ask in natural language. The model parses intent and "
                              "calls the right module.")
        self._clear_form()

        ttk.Label(self.form, text="Question:",
                  background="#f4f6fa").pack(side="left", padx=4)
        self.m4_input = tk.StringVar()
        entry = ttk.Entry(self.form, textvariable=self.m4_input, width=70)
        entry.pack(side="left", padx=4, fill="x", expand=True)
        entry.bind("<Return>", lambda e: self._m4_ask())
        ttk.Button(self.form, text="Ask",
                   command=self._m4_ask).pack(side="left", padx=4)
        ttk.Button(self.form, text="Help",
                   command=self._m4_help).pack(side="left", padx=4)

    def _m4_help(self):
        self._log(
            "\nExample questions:\n"
            "  • Top 10 pickup zones\n"
            "  • What's the peak hour on weekends?\n"
            "  • Predict demand for zone 132 at 8am Monday\n"
            "  • Estimated fare for 10 miles in rush hour\n"
            "  • Average speed at 3am\n"
            "  • Regenerate all charts\n"
        )

    def _m4_ask(self):
        if not self._ensure_data(): return
        q = self.m4_input.get().strip()
        if not q:
            return
        self._log(f"\nYou> {q}")
        self.m4_input.set("")
        self._set_status("Parsing question...")

        def task():
            try:
                parsed = llm_parse(q)
                intent = parsed.get("intent", "unknown")
                params = parsed.get("params", {})
                self._log(f"[parsed] intent={intent}  params={params}")

                if intent not in HANDLERS:
                    self._log("Bot> Sorry, I don't understand. Try 'Help'.")
                    self._set_status("Unknown intent"); return

                before = set(glob.glob(os.path.join(OUT_DIR, "*.png")))
                answer, chart = HANDLERS[intent](self.df, params)
                self._log(f"Bot> {answer}")
                self._log(f"     chart: {chart}")

                new = list_new_images(before)
                first = new[0] if new else (chart if str(chart).endswith(".png") else None)
                if first and os.path.exists(first):
                    self.show_image(first)
                    open_file_external(first)
                self._set_status(f"Answered ({intent})")
            except Exception as e:
                self._log(f"[error] {e}")
                self._set_status("Q&A failed")
        threading.Thread(target=task, daemon=True).start()

    # ---------- exit ----------
    def on_exit(self):
        if messagebox.askokcancel("Exit", "Close the application?"):
            self.destroy()

    # ---------- dispatcher (kept for completeness) ----------
    def _render_m1(self): self.show_m1()
    def _render_m2(self): self.show_m2()
    def _render_m3(self): self.show_m3()
    def _render_m4(self): self.show_m4()


# ============ Entry ============
if __name__ == "__main__":
    app = TaxiApp()
    app.protocol("WM_DELETE_WINDOW", app.on_exit)
    app.mainloop()