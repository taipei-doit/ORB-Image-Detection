import cv2
import numpy as np
import os
import sys
import shutil
import ctypes
import pickle
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

CACHE_FILENAME = ".orb_cache.dat"

# 修正 Windows 高 DPI 螢幕模糊問題
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-Monitor DPI Aware
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

class V8Engine:
    """核心比對引擎：結合 ORB 與 AKAZE，支援多重 RANSAC 模型偵測"""
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=5000)
        self.akaze = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)  # 共用，不重複建立

    def extract(self, img):
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self.orb.detectAndCompute(gray, None)
        kp2, des2 = self.akaze.detectAndCompute(gray, None)
        return {"orb": (kp1, des1), "akaze": (kp2, des2)}

    def match(self, q_des, t_des):
        if q_des is None or t_des is None or len(t_des) < 10: return []
        try:
            matches = self.bf.knnMatch(q_des, t_des, k=2)
            return [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        except:
            return []

    def multi_ransac(self, src, dst, max_models=3):
        models = []
        src_work, dst_work = src.copy(), dst.copy()
        for _ in range(max_models):
            if len(src_work) < 10: break
            M, mask = cv2.findHomography(src_work, dst_work, cv2.RANSAC, 5.0)
            if mask is None: break
            inliers_mask = mask.ravel().astype(bool)
            num_inliers = int(np.sum(inliers_mask))
            if num_inliers < 10: break
            models.append(num_inliers)
            src_work = src_work[~inliers_mask]
            dst_work = dst_work[~inliers_mask]
        return models

class ImageDetectionV83:
    def __init__(self, root):
        self.root = root
        self.root.title("圖片對比檢測工具")
        self.root.geometry("1200x900")
        self.engine = V8Engine()

        self.query_path = ""
        self.folder_path = ""
        self.folder_a_path = ""
        self.folder_b_path = ""

        # 特徵快取
        self._cached_folder = None
        self._cached_features = {}
        self._cached_folder_a = None
        self._cached_features_a = {}
        self._cached_folder_b = None
        self._cached_features_b = {}

        # N:M 結果暫存 (供篩選與熱點圖使用)
        self._nm_results = []

        # 中文字型設定
        self.chinese_font = ("Microsoft JhengHei", 10)
        self.chinese_font_bold = ("Microsoft JhengHei", 10, "bold")

        # ===== UI 介面 =====
        tk.Label(root, text="圖片對比檢測工具", font=("Microsoft JhengHei", 18, "bold")).pack(pady=10)

        # --- 模式切換 ---
        mode_frame = tk.Frame(root)
        mode_frame.pack(pady=5)
        self.mode_var = tk.StringVar(value="1:N")
        tk.Label(mode_frame, text="比對模式：", font=self.chinese_font_bold).pack(side="left", padx=(0, 5))
        tk.Radiobutton(mode_frame, text="1:N（單圖 vs 資料夾）", variable=self.mode_var,
                        value="1:N", command=self._on_mode_change, font=self.chinese_font).pack(side="left", padx=5)
        tk.Radiobutton(mode_frame, text="N:M（資料夾 vs 資料夾）", variable=self.mode_var,
                        value="N:M", command=self._on_mode_change, font=self.chinese_font).pack(side="left", padx=5)

        # --- 1:N 面板 ---
        self.panel_1n = tk.Frame(root)
        self.panel_1n.pack(pady=5)
        tk.Button(self.panel_1n, text="1. 選擇基準原圖", command=self.select_query,
                  width=15, font=self.chinese_font).grid(row=0, column=0, padx=5)
        tk.Button(self.panel_1n, text="2. 選擇比對資料夾", command=self.select_folder,
                  width=15, font=self.chinese_font).grid(row=0, column=1, padx=5)
        self.btn_run_1n = tk.Button(self.panel_1n, text="3. 執行檢索", command=self.run_batch,
                                     bg="#1B5E20", fg="white", font=self.chinese_font_bold, state="disabled")
        self.btn_run_1n.grid(row=0, column=2, padx=20)

        self.lbl_query = tk.Label(self.panel_1n, text="基準圖：未選擇", font=self.chinese_font, fg="gray")
        self.lbl_query.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self.lbl_folder = tk.Label(self.panel_1n, text="資料夾：未選擇", font=self.chinese_font, fg="gray")
        self.lbl_folder.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # --- N:M 面板 ---
        self.panel_nm = tk.Frame(root)
        tk.Button(self.panel_nm, text="1. 選擇資料夾 A", command=self.select_folder_a,
                  width=15, font=self.chinese_font).grid(row=0, column=0, padx=5)
        tk.Button(self.panel_nm, text="2. 選擇資料夾 B", command=self.select_folder_b,
                  width=15, font=self.chinese_font).grid(row=0, column=1, padx=5)
        self.btn_run_nm = tk.Button(self.panel_nm, text="3. 執行 N:M 比對", command=self.run_batch_nm,
                                     bg="#0D47A1", fg="white", font=self.chinese_font_bold, state="disabled")
        self.btn_run_nm.grid(row=0, column=2, padx=20)

        self.lbl_folder_a = tk.Label(self.panel_nm, text="資料夾 A：未選擇", font=self.chinese_font, fg="gray")
        self.lbl_folder_a.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self.lbl_folder_b = tk.Label(self.panel_nm, text="資料夾 B：未選擇", font=self.chinese_font, fg="gray")
        self.lbl_folder_b.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # --- 熱點圖按鈕 ---
        tk.Button(root, text="查看特徵熱點圖 (分析圖片細節對應)",
                  command=self.visualize_heatmap, bg="#D32F2F", fg="white",
                  font=("Microsoft JhengHei", 11, "bold"), height=2).pack(pady=8)

        # --- 進度條 ---
        self.progress = ttk.Progressbar(root, orient="horizontal", length=1000, mode="determinate")
        self.progress.pack(pady=3)
        self.lbl_progress = tk.Label(root, text="", font=self.chinese_font, fg="#555")
        self.lbl_progress.pack()

        # --- N:M 篩選列 ---
        self.filter_frame = tk.Frame(root)
        tk.Label(self.filter_frame, text="篩選來源圖：", font=self.chinese_font).pack(side="left", padx=(0, 5))
        self.filter_var = tk.StringVar(value="全部顯示")
        self.filter_combo = ttk.Combobox(self.filter_frame, textvariable=self.filter_var,
                                          state="readonly", width=40, font=self.chinese_font)
        self.filter_combo.pack(side="left", padx=5)
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        tk.Label(self.filter_frame, text="最低相似度：", font=self.chinese_font).pack(side="left", padx=(15, 5))
        self.threshold_var = tk.StringVar(value="全部")
        self.threshold_combo = ttk.Combobox(self.filter_frame, textvariable=self.threshold_var,
                                             state="readonly", width=12, font=self.chinese_font,
                                             values=["全部", "中度以上(>30%)", "高度以上(>60%)", "極高(>85%)"])
        self.threshold_combo.pack(side="left", padx=5)
        self.threshold_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        # --- Treeview ---
        style = ttk.Style()
        style.configure("Treeview", font=self.chinese_font, rowheight=28)
        style.configure("Treeview.Heading", font=self.chinese_font_bold)

        tree_frame = tk.Frame(root)
        tree_frame.pack(fill="both", expand=True, padx=20, pady=8)

        # 1:N 欄位
        self.cols_1n = ("Rank", "Filename", "Inliers", "Confidence", "Level", "Status")
        # N:M 欄位
        self.cols_nm = ("Rank", "Source", "Target", "Inliers", "Confidence", "Level", "Status")

        self.tree = ttk.Treeview(tree_frame, columns=self.cols_1n, show="headings")
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._setup_tree_columns_1n()

        # 預設顯示 1:N 面板
        self._on_mode_change()

    # ===================== 模式切換 =====================
    def _on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "1:N":
            self.panel_nm.pack_forget()
            self.filter_frame.pack_forget()
            self.panel_1n.pack(pady=5, after=self.root.winfo_children()[1])  # after mode_frame
            self._setup_tree_columns_1n()
        else:
            self.panel_1n.pack_forget()
            self.panel_nm.pack(pady=5, after=self.root.winfo_children()[1])
            self.filter_frame.pack(pady=3, before=self.tree.master)
            self._setup_tree_columns_nm()

    def _setup_tree_columns_1n(self):
        self.tree["columns"] = self.cols_1n
        self.tree.delete(*self.tree.get_children())
        for c in self.cols_1n:
            self.tree.heading(c, text=c)
        self.tree.column("Rank", width=60, anchor="center")
        self.tree.column("Filename", width=400)
        self.tree.column("Inliers", width=100, anchor="center")
        self.tree.column("Confidence", width=100, anchor="center")
        self.tree.column("Level", width=150, anchor="center")
        self.tree.column("Status", width=80, anchor="center")

    def _setup_tree_columns_nm(self):
        self.tree["columns"] = self.cols_nm
        self.tree.delete(*self.tree.get_children())
        for c in self.cols_nm:
            self.tree.heading(c, text=c)
        self.tree.column("Rank", width=50, anchor="center")
        self.tree.column("Source", width=250)
        self.tree.column("Target", width=250)
        self.tree.column("Inliers", width=80, anchor="center")
        self.tree.column("Confidence", width=90, anchor="center")
        self.tree.column("Level", width=120, anchor="center")
        self.tree.column("Status", width=70, anchor="center")

    # ===================== 1:N 選擇器 =====================
    def select_query(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.webp")])
        if path:
            self.query_path = path
            self.lbl_query.config(text=f"基準圖：{os.path.basename(path)}", fg="#1B5E20")
            self._update_1n_button()

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_path = path
            self.lbl_folder.config(text=f"資料夾：{path}", fg="#1B5E20")
            self._update_1n_button()

    def _update_1n_button(self):
        self.btn_run_1n.config(state="normal" if self.query_path and self.folder_path else "disabled")

    # ===================== N:M 選擇器 =====================
    def select_folder_a(self):
        path = filedialog.askdirectory(title="選擇資料夾 A（來源圖片）")
        if path:
            self.folder_a_path = path
            count = self._count_images(path)
            self.lbl_folder_a.config(text=f"資料夾 A：{path}（{count} 張圖片）", fg="#0D47A1")
            self._update_nm_button()

    def select_folder_b(self):
        path = filedialog.askdirectory(title="選擇資料夾 B（比對目標）")
        if path:
            self.folder_b_path = path
            count = self._count_images(path)
            self.lbl_folder_b.config(text=f"資料夾 B：{path}（{count} 張圖片）", fg="#0D47A1")
            self._update_nm_button()

    def _update_nm_button(self):
        self.btn_run_nm.config(state="normal" if self.folder_a_path and self.folder_b_path else "disabled")

    @staticmethod
    def _count_images(folder):
        exts = ('.jpg', '.png', '.jpeg', '.webp')
        return sum(1 for f in os.listdir(folder) if f.lower().endswith(exts))

    @staticmethod
    def _list_images(folder):
        exts = ('.jpg', '.png', '.jpeg', '.webp')
        return [f for f in os.listdir(folder) if f.lower().endswith(exts)]

    # ===================== 共用工具 =====================
    def read_img(self, path):
        """讀取支援中文路徑的圖片檔案"""
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def _kp_to_tuple(kp_list):
        return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp_list]

    @staticmethod
    def _tuple_to_kp(tuple_list):
        return [cv2.KeyPoint(x=t[0][0], y=t[0][1], size=t[1], angle=t[2],
                             response=t[3], octave=t[4], class_id=t[5]) for t in tuple_list]

    def _serialize_features(self, data):
        if data is None:
            return None
        return {
            "orb": (self._kp_to_tuple(data["orb"][0]), data["orb"][1]),
            "akaze": (self._kp_to_tuple(data["akaze"][0]), data["akaze"][1]),
        }

    def _deserialize_features(self, data):
        if data is None:
            return None
        return {
            "orb": (self._tuple_to_kp(data["orb"][0]), data["orb"][1]),
            "akaze": (self._tuple_to_kp(data["akaze"][0]), data["akaze"][1]),
        }

    def _get_cache_path(self, folder):
        return os.path.join(folder, CACHE_FILENAME)

    def _load_disk_cache(self, folder):
        cache_path = self._get_cache_path(folder)
        if not os.path.exists(cache_path):
            return {}
        try:
            with open(cache_path, "rb") as f:
                raw = pickle.load(f)
            result = {}
            for filename, entry in raw.items():
                flips = {}
                for flip_key, sdata in entry["flips"].items():
                    flips[flip_key] = self._deserialize_features(sdata)
                result[filename] = {"path": entry["path"], "flips": flips}
            return result
        except Exception:
            return {}

    def _save_disk_cache(self, folder, features):
        cache_path = self._get_cache_path(folder)
        raw = {}
        for filename, entry in features.items():
            flips = {}
            for flip_key, fdata in entry["flips"].items():
                flips[flip_key] = self._serialize_features(fdata)
            raw[filename] = {"path": entry["path"], "flips": flips}
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(raw, f)
        except Exception:
            pass

    def _extract_features_for_folder(self, folder, cached_ref, files, progress_offset=0, progress_label=""):
        """提取指定檔案的特徵 (含翻轉)，結果加入快取"""
        for i, filename in enumerate(files):
            t_path = os.path.join(folder, filename)
            t_img = self.read_img(t_path)
            if t_img is None:
                continue

            flips_data = {}
            for flip in [None, 1, 0, -1]:
                img = t_img if flip is None else cv2.flip(t_img, flip)
                t_low = cv2.resize(img, (1000, int(1000 * img.shape[0] / img.shape[1])))
                t_data = self.engine.extract(t_low)
                flips_data[flip] = t_data

            cached_ref[filename] = {"path": t_path, "flips": flips_data}
            self.progress["value"] = progress_offset + i + 1
            if progress_label:
                self.lbl_progress.config(text=f"{progress_label}（{i+1}/{len(files)}）")
            self.root.update_idletasks()

    def _ensure_folder_cache(self, folder, cached_folder_attr, cached_features_attr,
                              progress_offset=0, progress_label=""):
        """確保某資料夾的特徵已快取，回傳快取字典"""
        cached_folder = getattr(self, cached_folder_attr)
        cached_features = getattr(self, cached_features_attr)

        if cached_folder != folder:
            setattr(self, cached_folder_attr, folder)
            cached_features = self._load_disk_cache(folder)
            setattr(self, cached_features_attr, cached_features)

        files = self._list_images(folder)
        new_files = [f for f in files if f not in cached_features]
        if new_files:
            self._extract_features_for_folder(folder, cached_features, new_files,
                                               progress_offset, progress_label)
            self._save_disk_cache(folder, cached_features)

        return cached_features, files

    # ===================== 特徵比對 =====================
    def _match_single_alg(self, q_data, t_data, alg):
        m = self.engine.match(q_data[alg][1], t_data[alg][1])
        if len(m) < 10: return 0
        src_pts = np.float32([q_data[alg][0][x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
        dst_pts = np.float32([t_data[alg][0][x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
        models = self.engine.multi_ransac(src_pts, dst_pts)
        return sum(models) if models else 0

    def process_compare(self, q_data, t_data):
        """比對兩組已提取的特徵資料 (ORB 優先，低於門檻才補跑 AKAZE)"""
        if not t_data: return 0, 0
        q_orb_count = len(q_data["orb"][0]) if q_data["orb"][0] else 0
        if q_orb_count == 0: return 0, 0

        orb_in = self._match_single_alg(q_data, t_data, "orb")
        orb_conf = min((orb_in / q_orb_count) * 100, 100.0)

        if orb_conf < 1:
            return orb_in, orb_conf

        akaze_in = self._match_single_alg(q_data, t_data, "akaze")
        total_in = orb_in + akaze_in
        conf = min((total_in / q_orb_count) * 100, 100.0)
        return total_in, conf

    def _compare_with_flips(self, q_data, target_cache_entry):
        """對一組 query 特徵 vs 一張目標圖的所有翻轉做比對，回傳最佳結果"""
        best_conf, best_in = -1, 0
        for flip, t_data in target_cache_entry["flips"].items():
            if t_data is None:
                continue
            inliers, conf = self.process_compare(q_data, t_data)
            if conf > best_conf:
                best_conf, best_in = conf, inliers
            if best_conf > 85:
                break
        return best_in, best_conf

    @staticmethod
    def _conf_to_level(conf):
        if conf > 85: return "極高相似"
        if conf > 60: return "高度相似"
        if conf > 30: return "中度相似"
        return "無關"

    # ===================== 1:N 批次比對 =====================
    def run_batch(self):
        query_fn = os.path.splitext(os.path.basename(self.query_path))[0]
        task_dir = os.path.join(self.folder_path, f"V83_Result_{query_fn}")

        q_img = self.read_img(self.query_path)
        q_low = cv2.resize(q_img, (1000, int(1000 * q_img.shape[0] / q_img.shape[1])))
        q_data = self.engine.extract(q_low)

        files = self._list_images(self.folder_path)

        # 快取管理（向下相容舊的 1:N 快取屬性）
        if self._cached_folder != self.folder_path:
            self._cached_folder = self.folder_path
            self._cached_features = self._load_disk_cache(self.folder_path)

        new_files = [f for f in files if f not in self._cached_features]
        if new_files:
            self.progress["maximum"] = len(new_files)
            self.lbl_progress.config(text="提取特徵中...")
            self._extract_features_for_folder(self.folder_path, self._cached_features, new_files,
                                               progress_label="提取特徵")
            self._save_disk_cache(self.folder_path, self._cached_features)

        # 比對階段
        self.progress["value"] = 0
        self.progress["maximum"] = len(files)
        results = []

        for i, filename in enumerate(files):
            if filename not in self._cached_features:
                continue
            cache = self._cached_features[filename]
            best_in, best_conf = self._compare_with_flips(q_data, cache)
            level = self._conf_to_level(best_conf)
            results.append({"name": filename, "path": cache["path"],
                            "in": best_in, "conf": best_conf, "level": level})
            self.progress["value"] = i + 1
            self.lbl_progress.config(text=f"比對中（{i+1}/{len(files)}）")
            self.root.update_idletasks()

        results.sort(key=lambda x: x['conf'], reverse=True)
        self.tree.delete(*self.tree.get_children())
        if not os.path.exists(task_dir): os.makedirs(task_dir)

        for idx, r in enumerate(results):
            is_extracted = r['conf'] >= 30
            status = "已提取" if is_extracted else "-"
            if is_extracted: shutil.copy(r['path'], os.path.join(task_dir, r['name']))
            self.tree.insert("", "end", values=(idx+1, r['name'], r['in'],
                             f"{r['conf']:.2f}%", r['level'], status))

        self.lbl_progress.config(text=f"完成！共比對 {len(results)} 張圖片")
        messagebox.showinfo("完成", f"分析報告已生成至資料夾：\n{task_dir}")

    # ===================== N:M 批次比對 =====================
    def run_batch_nm(self):
        files_a = self._list_images(self.folder_a_path)
        files_b = self._list_images(self.folder_b_path)

        if not files_a or not files_b:
            messagebox.showwarning("提示", "兩個資料夾都必須包含圖片！")
            return

        total_extract = len(files_a) + len(files_b)
        total_compare = len(files_a) * len(files_b)

        # 確認大量比對
        if total_compare > 5000:
            ok = messagebox.askyesno("確認",
                f"即將進行 {len(files_a)} × {len(files_b)} = {total_compare:,} 次比對。\n"
                f"這可能需要較長時間，是否繼續？")
            if not ok:
                return

        # 停用按鈕避免重複點擊
        self.btn_run_nm.config(state="disabled")

        # === 階段一：提取特徵 ===
        self.progress["value"] = 0
        self.progress["maximum"] = total_extract
        self.lbl_progress.config(text="提取資料夾 A 特徵...")
        self.root.update_idletasks()

        cache_a, files_a = self._ensure_folder_cache(
            self.folder_a_path, "_cached_folder_a", "_cached_features_a",
            progress_offset=0, progress_label="資料夾 A 特徵提取")

        self.lbl_progress.config(text="提取資料夾 B 特徵...")
        self.root.update_idletasks()

        cache_b, files_b = self._ensure_folder_cache(
            self.folder_b_path, "_cached_folder_b", "_cached_features_b",
            progress_offset=len(files_a), progress_label="資料夾 B 特徵提取")

        # === 階段二：N×M 比對 ===
        self.progress["value"] = 0
        self.progress["maximum"] = total_compare
        self.lbl_progress.config(text=f"N:M 比對中（0/{total_compare:,}）")
        self.root.update_idletasks()

        results = []
        count = 0

        for fa in files_a:
            if fa not in cache_a:
                continue
            entry_a = cache_a[fa]
            # 對資料夾 A 的圖片，取原始特徵 (flip=None) 作為 query
            q_data = entry_a["flips"].get(None)
            if q_data is None:
                count += len(files_b)
                continue

            for fb in files_b:
                if fb not in cache_b:
                    count += 1
                    continue
                entry_b = cache_b[fb]
                best_in, best_conf = self._compare_with_flips(q_data, entry_b)
                level = self._conf_to_level(best_conf)
                results.append({
                    "source": fa, "target": fb,
                    "source_path": entry_a["path"], "target_path": entry_b["path"],
                    "in": best_in, "conf": best_conf, "level": level
                })
                count += 1

                # 每 50 次更新一次 UI，避免過度刷新
                if count % 50 == 0:
                    self.progress["value"] = count
                    self.lbl_progress.config(text=f"N:M 比對中（{count:,}/{total_compare:,}）")
                    self.root.update_idletasks()

        self.progress["value"] = total_compare

        # 儲存結果供篩選使用
        results.sort(key=lambda x: x['conf'], reverse=True)
        self._nm_results = results

        # 建立結果資料夾並複製相似圖片
        dir_a_name = os.path.basename(self.folder_a_path)
        dir_b_name = os.path.basename(self.folder_b_path)
        task_dir = os.path.join(self.folder_b_path, f"NM_Result_{dir_a_name}_vs_{dir_b_name}")
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        extracted_count = 0
        for r in results:
            if r['conf'] >= 30:
                # 建立以來源圖命名的子資料夾
                src_name = os.path.splitext(r['source'])[0]
                sub_dir = os.path.join(task_dir, src_name)
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                shutil.copy(r['target_path'], os.path.join(sub_dir, r['target']))
                extracted_count += 1

        # 更新篩選下拉選單
        sources = sorted(set(r['source'] for r in results))
        self.filter_combo["values"] = ["全部顯示"] + sources
        self.filter_var.set("全部顯示")
        self.threshold_var.set("全部")

        # 顯示結果
        self._display_nm_results()

        self.lbl_progress.config(text=f"完成！{len(files_a)} × {len(files_b)} = {total_compare:,} 次比對，"
                                      f"提取 {extracted_count} 組相似配對")
        self.btn_run_nm.config(state="normal")
        messagebox.showinfo("完成",
            f"N:M 比對完成！\n\n"
            f"來源：{len(files_a)} 張 × 目標：{len(files_b)} 張\n"
            f"總比對次數：{total_compare:,}\n"
            f"相似配對（>30%）：{extracted_count} 組\n\n"
            f"結果已生成至：\n{task_dir}")

    def _display_nm_results(self):
        """根據篩選條件顯示 N:M 結果"""
        self.tree.delete(*self.tree.get_children())

        source_filter = self.filter_var.get()
        threshold_str = self.threshold_var.get()

        # 解析門檻
        threshold = 0
        if "30" in threshold_str: threshold = 30
        elif "60" in threshold_str: threshold = 60
        elif "85" in threshold_str: threshold = 85

        filtered = self._nm_results
        if source_filter != "全部顯示":
            filtered = [r for r in filtered if r['source'] == source_filter]
        if threshold > 0:
            filtered = [r for r in filtered if r['conf'] >= threshold]

        for idx, r in enumerate(filtered):
            status = "已提取" if r['conf'] >= 30 else "-"
            self.tree.insert("", "end", values=(
                idx + 1, r['source'], r['target'], r['in'],
                f"{r['conf']:.2f}%", r['level'], status))

    def _on_filter_change(self, event=None):
        if self._nm_results:
            self._display_nm_results()

    # ===================== 熱點圖 =====================
    def generate_heatmap_layer(self, img, keypoints):
        heatmap = np.zeros(img.shape[:2], dtype=np.float32)
        for kp in keypoints:
            x, y = map(int, kp.pt)
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(heatmap, (x, y), 20, 1, -1)
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        cv2.normalize(heatmap, heatmap, 0, 1, cv2.NORM_MINMAX)
        heatmap = np.uint8(255 * heatmap)
        color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return cv2.addWeighted(img, 0.7, color_heatmap, 0.3, 0)

    def visualize_heatmap(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("提示", "請先執行檢索並從清單中選中一張圖片！")
            return

        item = self.tree.item(selected)['values']
        mode = self.mode_var.get()

        if mode == "1:N":
            if not self.query_path:
                messagebox.showwarning("提示", "請先選擇基準原圖！")
                return
            q_path = self.query_path
            t_path = os.path.join(self.folder_path, item[1])
        else:
            # N:M 模式：Source 在 item[1], Target 在 item[2]
            source_name = item[1]
            target_name = item[2]
            q_path = os.path.join(self.folder_a_path, source_name)
            t_path = os.path.join(self.folder_b_path, target_name)

        q_img = self.read_img(q_path)
        t_img = self.read_img(t_path)
        if q_img is None or t_img is None:
            messagebox.showerror("錯誤", "無法讀取圖片檔案！")
            return

        display_h = 750
        q_w = int(q_img.shape[1] * (display_h / q_img.shape[0]))
        t_w = int(t_img.shape[1] * (display_h / t_img.shape[0]))

        q_vis = cv2.resize(q_img, (q_w, display_h))
        t_vis = cv2.resize(t_img, (t_w, display_h))

        q_d = self.engine.extract(q_vis)
        t_d = self.engine.extract(t_vis)
        matches = self.engine.match(q_d["akaze"][1], t_d["akaze"][1])
        matches = sorted(matches, key=lambda x: x.distance)[:70]

        t_kp_matched = [t_d["akaze"][0][m.trainIdx] for m in matches]
        t_heatmap_overlay = self.generate_heatmap_layer(t_vis, t_kp_matched)

        canvas = np.zeros((display_h, q_w + t_w, 3), dtype=np.uint8)
        canvas[:, :q_w] = q_vis
        canvas[:, q_w:] = t_heatmap_overlay

        for m in matches:
            p1 = tuple(map(int, q_d["akaze"][0][m.queryIdx].pt))
            p2 = tuple(map(int, t_d["akaze"][0][m.trainIdx].pt))
            p2_with_offset = (p2[0] + q_w, p2[1])
            cv2.line(canvas, p1, p2_with_offset, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, (0, 0, 255), -1)
            cv2.circle(canvas, p2_with_offset, 3, (0, 0, 255), -1)

        win_title = "Heatmap Analysis"
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.imshow(win_title, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call('tk', 'scaling', root.winfo_fpixels('1i') / 72.0)
    except Exception:
        pass
    app = ImageDetectionV83(root)
    root.mainloop()
