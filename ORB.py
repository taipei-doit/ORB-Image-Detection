import cv2
import numpy as np
import os
import sys
import shutil
import ctypes
import json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

CACHE_FILENAME = ".orb_cache.json"

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
        self.root.geometry("1150x850")
        self.engine = V8Engine()
        
        self.query_path = ""
        self.folder_path = ""

        # 特徵快取 (同一資料夾只提取一次，換基準圖不需重新計算)
        self._cached_folder = None
        self._cached_features = {}  # {filename: {"data": feature_data, "flips": {flip: feature_data}}}

        # 中文字型設定 (解決中文檔名亂碼問題)
        self.chinese_font = ("Microsoft JhengHei", 10)
        self.chinese_font_bold = ("Microsoft JhengHei", 10, "bold")

        # UI 介面設定
        tk.Label(root, text="圖片對比檢測工具", font=("Microsoft JhengHei", 18, "bold")).pack(pady=15)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="1. 選擇基準原圖", command=self.select_query, width=15, font=self.chinese_font).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="2. 選擇比對資料夾", command=self.select_folder, width=15, font=self.chinese_font).grid(row=0, column=1, padx=5)
        self.btn_run = tk.Button(btn_frame, text="3. 執行檢索", command=self.run_batch,
                                 bg="#1B5E20", fg="white", font=("Microsoft JhengHei", 10, "bold"), state="disabled")
        self.btn_run.grid(row=0, column=2, padx=20)

        # 視覺化熱點按鈕
        tk.Button(root, text="查看特徵熱點圖 (分析圖片細節對應)",
                  command=self.visualize_heatmap, bg="#D32F2F", fg="white",
                  font=("Microsoft JhengHei", 11, "bold"), height=2).pack(pady=10)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=1000, mode="determinate")
        self.progress.pack(pady=5)

        # 設定 Treeview 字型以支援中文顯示
        style = ttk.Style()
        style.configure("Treeview", font=self.chinese_font, rowheight=28)
        style.configure("Treeview.Heading", font=self.chinese_font_bold)

        cols = ("Rank", "Filename", "Inliers", "Confidence", "Level", "Status")
        self.tree = ttk.Treeview(root, columns=cols, show="headings")
        for c in cols: self.tree.heading(c, text=c)
        self.tree.column("Rank", width=60, anchor="center")
        self.tree.column("Filename", width=400)
        self.tree.column("Inliers", width=100, anchor="center")
        self.tree.column("Confidence", width=100, anchor="center")
        self.tree.column("Level", width=150, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=20, pady=10)

    def select_query(self):
        self.query_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.webp")])
        if self.query_path: self.btn_run.config(state="normal" if self.folder_path else "disabled")

    def select_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path: self.btn_run.config(state="normal" if self.query_path else "disabled")

    def read_img(self, path):
        """讀取支援中文路徑的圖片檔案"""
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def _kp_to_tuple(kp_list):
        """將 cv2.KeyPoint 列表轉成可序列化的 tuple 列表"""
        return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp_list]

    @staticmethod
    def _tuple_to_kp(tuple_list):
        """將 tuple 列表還原成 cv2.KeyPoint 列表"""
        return [cv2.KeyPoint(x=t[0][0], y=t[0][1], size=t[1], angle=t[2],
                             response=t[3], octave=t[4], class_id=t[5]) for t in tuple_list]

    def _serialize_features(self, data):
        """將特徵資料轉成可存檔格式"""
        if data is None:
            return None
        return {
            "orb": (self._kp_to_tuple(data["orb"][0]), data["orb"][1].tolist() if data["orb"][1] is not None else None),
            "akaze": (self._kp_to_tuple(data["akaze"][0]), data["akaze"][1].tolist() if data["akaze"][1] is not None else None),
        }

    def _deserialize_features(self, data):
        """將存檔格式還原成特徵資料"""
        if data is None:
            return None
        return {
            "orb": (self._tuple_to_kp(data["orb"][0]), np.array(data["orb"][1], dtype=np.uint8) if data["orb"][1] is not None else None),
            "akaze": (self._tuple_to_kp(data["akaze"][0]), np.array(data["akaze"][1], dtype=np.uint8) if data["akaze"][1] is not None else None),
        }

    def _get_cache_path(self):
        return os.path.join(self.folder_path, CACHE_FILENAME)

    def _load_disk_cache(self):
        """從資料夾讀取磁碟快取"""
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return {}
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # 還原 KeyPoint 物件
            result = {}
            for filename, entry in raw.items():
                flips = {}
                for flip_key, sdata in entry["flips"].items():
                    flips[flip_key] = self._deserialize_features(sdata)
                result[filename] = {"path": entry["path"], "flips": flips}
            return result
        except Exception:
            return {}

    def _save_disk_cache(self):
        """將快取寫入資料夾"""
        cache_path = self._get_cache_path()
        raw = {}
        for filename, entry in self._cached_features.items():
            flips = {}
            for flip_key, fdata in entry["flips"].items():
                flips[flip_key] = self._serialize_features(fdata)
            raw[filename] = {"path": entry["path"], "flips": flips}
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(raw, f)
        except Exception:
            pass

    def generate_heatmap_layer(self, img, keypoints):
        """生成熱點覆蓋層"""
        heatmap = np.zeros(img.shape[:2], dtype=np.float32)
        for kp in keypoints:
            x, y = map(int, kp.pt)
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(heatmap, (x, y), 20, 1, -1)
        
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        cv2.normalize(heatmap, heatmap, 0, 1, cv2.NORM_MINMAX)
        heatmap = np.uint8(255 * heatmap)
        color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # 0.7 原圖 + 0.3 熱力圖，增加對比度
        return cv2.addWeighted(img, 0.7, color_heatmap, 0.3, 0)

    def visualize_heatmap(self):
        """核心修正：動態畫布拼接，確保顯示不完整問題完全解決"""
        selected = self.tree.selection()
        if not selected or not self.query_path:
            messagebox.showwarning("提示", "請先執行檢索並從清單中選中一張圖片！")
            return
        
        item = self.tree.item(selected)['values']
        filename = item[1]
        t_path = os.path.join(self.folder_path, filename)

        q_img = self.read_img(self.query_path)
        t_img = self.read_img(t_path)
        if q_img is None or t_img is None: return

        # 1. 統一顯示高度 (固定為 750px，寬度隨比例縮放)
        display_h = 750
        q_w = int(q_img.shape[1] * (display_h / q_img.shape[0]))
        t_w = int(t_img.shape[1] * (display_h / t_img.shape[0]))
        
        q_vis = cv2.resize(q_img, (q_w, display_h))
        t_vis = cv2.resize(t_img, (t_w, display_h))

        # 2. 提取特徵與匹配 (使用細節更精確的 AKAZE)
        q_d = self.engine.extract(q_vis)
        t_d = self.engine.extract(t_vis)
        matches = self.engine.match(q_d["akaze"][1], t_d["akaze"][1])
        matches = sorted(matches, key=lambda x: x.distance)[:70] # 僅取最準的前70個點

        # 3. 準備熱點圖
        t_kp_matched = [t_d["akaze"][0][m.trainIdx] for m in matches]
        t_heatmap_overlay = self.generate_heatmap_layer(t_vis, t_kp_matched)

        # 4. 建立完整拼接畫布 (解決顯示不完整問題)
        canvas = np.zeros((display_h, q_w + t_w, 3), dtype=np.uint8)
        canvas[:, :q_w] = q_vis
        canvas[:, q_w:] = t_heatmap_overlay

        # 5. 手動繪製連線與關鍵點
        for m in matches:
            p1 = tuple(map(int, q_d["akaze"][0][m.queryIdx].pt))
            p2 = tuple(map(int, t_d["akaze"][0][m.trainIdx].pt))
            p2_with_offset = (p2[0] + q_w, p2[1])
            
            # 綠色連線 (指出精確位置)
            cv2.line(canvas, p1, p2_with_offset, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, (0, 0, 255), -1)
            cv2.circle(canvas, p2_with_offset, 3, (0, 0, 255), -1)

        # 顯示視窗 (使用英文視窗標題避免 OpenCV 中文亂碼)
        win_title = "Heatmap Analysis"
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.imshow(win_title, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _build_cache(self, files):
        """提取指定檔案的特徵 (含翻轉)，結果加入快取"""
        self.progress["maximum"] = len(files)

        for i, filename in enumerate(files):
            t_path = os.path.join(self.folder_path, filename)
            t_img = self.read_img(t_path)
            if t_img is None:
                continue

            flips_data = {}
            for flip in [None, 1, 0, -1]:
                img = t_img if flip is None else cv2.flip(t_img, flip)
                t_low = cv2.resize(img, (1000, int(1000 * img.shape[0] / img.shape[1])))
                t_data = self.engine.extract(t_low)
                flips_data[flip] = t_data

            self._cached_features[filename] = {"path": t_path, "flips": flips_data}
            self.progress["value"] = i + 1
            self.root.update_idletasks()

        # 提取完畢後自動存檔到磁碟
        self._save_disk_cache()

    def run_batch(self):
        """執行批次比對與提取"""
        query_fn = os.path.splitext(os.path.basename(self.query_path))[0]
        task_dir = os.path.join(self.folder_path, f"V83_Result_{query_fn}")

        q_img = self.read_img(self.query_path)
        q_low = cv2.resize(q_img, (1000, int(1000 * q_img.shape[0] / q_img.shape[1])))
        q_data = self.engine.extract(q_low)

        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]

        # 換資料夾 → 嘗試讀取磁碟快取；同資料夾 → 沿用記憶體快取
        if self._cached_folder != self.folder_path:
            self._cached_folder = self.folder_path
            self._cached_features = self._load_disk_cache()

        # 找出尚未快取的檔案，只提取這些
        new_files = [f for f in files if f not in self._cached_features]
        if new_files:
            self._build_cache(new_files)

        # 比對階段 (使用快取 + 提前跳出優化)
        self.progress["maximum"] = len(files)
        results = []

        for i, filename in enumerate(files):
            if filename not in self._cached_features:
                continue

            cache = self._cached_features[filename]
            best_conf, best_in = -1, 0

            for flip, t_data in cache["flips"].items():
                if t_data is None:
                    continue
                inliers, conf = self.process_compare(q_data, t_data)
                if conf > best_conf:
                    best_conf, best_in = conf, inliers
                # 已經極高相似，不需要再試其他翻轉
                if best_conf > 85:
                    break

            level = "極高相似" if best_conf > 85 else "高度相似" if best_conf > 60 else "中度相似" if best_conf > 30 else "無關"
            results.append({"name": filename, "path": cache["path"], "in": best_in, "conf": best_conf, "level": level})
            self.progress["value"] = i + 1
            self.root.update_idletasks()

        results.sort(key=lambda x: x['conf'], reverse=True)
        self.tree.delete(*self.tree.get_children())
        if not os.path.exists(task_dir): os.makedirs(task_dir)

        for idx, r in enumerate(results):
            is_extracted = r['conf'] >= 30
            status = "已提取" if is_extracted else "-"
            if is_extracted: shutil.copy(r['path'], os.path.join(task_dir, r['name']))
            self.tree.insert("", "end", values=(idx+1, r['name'], r['in'], f"{r['conf']:.2f}%", r['level'], status))

        messagebox.showinfo("完成", f"分析報告已生成至資料夾：\n{task_dir}")

    def _match_single_alg(self, q_data, t_data, alg):
        """單一演算法的匹配 + RANSAC"""
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

        # 先跑 ORB (速度快)
        orb_in = self._match_single_alg(q_data, t_data, "orb")
        orb_conf = min((orb_in / q_orb_count) * 100, 100.0)

        # ORB 幾乎無匹配 (< 1%) 才跳過 AKAZE，避免漏掉二次截圖或局部裁切
        if orb_conf < 1:
            return orb_in, orb_conf

        # 補跑 AKAZE 提高精確度
        akaze_in = self._match_single_alg(q_data, t_data, "akaze")
        total_in = orb_in + akaze_in
        conf = min((total_in / q_orb_count) * 100, 100.0)
        return total_in, conf

if __name__ == "__main__":
    root = tk.Tk()
    # 讓 Tkinter 正確處理 DPI 縮放
    try:
        root.tk.call('tk', 'scaling', root.winfo_fpixels('1i') / 72.0)
    except Exception:
        pass
    app = ImageDetectionV83(root)
    root.mainloop()
