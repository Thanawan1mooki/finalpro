# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os

# # โหลดโมเดล YOLO ที่ถูก Fine-tuned แล้ว (best.pt)
# model = None
# try:
#     model = YOLO("best.pt")  # ตรวจสอบให้แน่ใจว่า best.pt อยู่โฟลเดอร์เดียวกับไฟล์นี้
#     print("Fine-tuned YOLO model loaded successfully.")
# except Exception as e:
#     print(f"Error loading fine-tuned YOLO model (best.pt): {e}")
#     print("Please ensure 'best.pt' is in the same directory as this script.")

# def detect_banana(image_path):
#     if model is None:
#         print("Error: YOLO model is not loaded. Cannot perform detection.")
#         return [], os.path.join("static", "error_processing.jpg")

#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not load image from {image_path}")
#         return [], os.path.join("static", "error_processing.jpg")

#     output_path = os.path.join("static", "result.jpg")

#     try:
#         results_list = model.predict(img, conf=0.5, iou=0.45, verbose=False)
#         results = results_list[0]
#         labels_info = []

#         if len(results.boxes) == 0:
#             print("No bananas found. Saving original image.")
#             cv2.imwrite(output_path, img)
#             return [], output_path

#         H_img, W_img = img.shape[:2]

#         for i, box in enumerate(results.boxes.data):
#             x1, y1, x2, y2 = map(int, box[:4])
#             conf = float(box[4])
#             cls_id = int(box[5])

#             # ป้องกันไม่ให้ออกนอกขอบภาพ
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(W_img - 1, x2), min(H_img - 1, y2)

#             # หดกรอบ ~5%
#             bw, bh = max(1, x2 - x1), max(1, y2 - y1)
#             padx, pady = max(1, int(0.05 * bw)), max(1, int(0.05 * bh))
#             xi1 = min(max(0, x1 + padx), W_img - 2)
#             yi1 = min(max(0, y1 + pady), H_img - 2)
#             xi2 = max(min(x2 - padx, W_img - 1), xi1 + 1)
#             yi2 = max(min(y2 - pady, H_img - 1), yi1 + 1)

#             crop = img[yi1:yi2, xi1:xi2]
#             if crop.size == 0:
#                 continue

#             # ============== BEGIN: HSV + spot detection block ==============
#             hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#             H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

#             base_mask = np.where((S > 25) | ((V < 230) & (V > 20)), 255, 0).astype(np.uint8)
#             base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), 1)
#             base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), 2)
#             m = base_mask > 0

#             brown = ((H >= 8) & (H <= 30) & (S >= 60) & (V <= 170) & m)
#             black = ((V <= 60) & m)

#             spot_kernel = np.ones((3,3), np.uint8)
#             brown_clean = cv2.morphologyEx((brown.astype(np.uint8) * 255), cv2.MORPH_OPEN, spot_kernel, iterations=1)
#             black_clean = cv2.morphologyEx((black.astype(np.uint8) * 255), cv2.MORPH_OPEN, spot_kernel, iterations=1)
#             spot_mask = cv2.bitwise_or(brown_clean, black_clean)

#             num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats((spot_mask>0).astype(np.uint8), connectivity=8)
#             spot_area = int((spot_mask>0).sum())
#             fg_area   = int(m.sum())
#             spot_frac = (spot_area / max(fg_area, 1))
#             spot_cnt  = int(((stats[1:, cv2.CC_STAT_AREA] >= 15) & (stats[1:, cv2.CC_STAT_AREA] <= 2000)).sum())

#             H_vals, S_vals, V_vals = H[m], S[m], V[m]
#             H_mean = float(np.mean(H_vals))   if H_vals.size else 0.0
#             H_med  = float(np.median(H_vals)) if H_vals.size else 0.0
#             S_mean = float(np.mean(S_vals))   if S_vals.size else 0.0
#             V_mean = float(np.mean(V_vals))   if V_vals.size else 0.0
#             V_std  = float(np.std(V_vals))    if V_vals.size else 0.0

#             dark_frac      = float(np.mean((V_vals < 80)))  if V_vals.size else 0.0
#             very_dark_frac = float(np.mean((V_vals < 50)))  if V_vals.size else 0.0
#             lowsat_frac    = float(np.mean((S_vals < 60)))  if V_vals.size else 0.0

# # 🔎 Debug print
#             print(f"Banana {i+1}: H_mean={H_mean:.1f}, H_med={H_med:.1f}, "
#             f"S_mean={S_mean:.1f}, V_mean={V_mean:.1f}, dark_frac={dark_frac:.2f}")           

#             edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150)
#             edge_density = float(np.mean((edges > 0) & m)) if edges.size else 0.0

#             label = model.names[cls_id]

#             is_really_rotten = (
#                 very_dark_frac >= 0.35 or
#                 (dark_frac >= 0.65 and lowsat_frac >= 0.60) or
#                 ((V_mean < 85 and S_mean < 70) and edge_density < 0.02)
#             )

#             is_spotty_overripe = (
#                 (spot_frac >= 0.035) or
#                 (spot_cnt  >= 8)    or
#                 (dark_frac >= 0.20 and edge_density >= 0.02)
#             )

#             if is_really_rotten:
#                 label = "rotten"
#             elif is_spotty_overripe:
#                 label = "overripe"
#             else:
#                 H_YELLOW_MIN, H_YELLOW_MAX = 15, 38
#                 H_GREEN_MIN,  H_GREEN_MAX  = 38, 85

#                 if H_GREEN_MIN <= H_med < 65 and S_mean > 40 and V_mean > 60:
#                     label = "unripe"
#                 elif 65 <= H_med <= H_GREEN_MAX and S_mean > 35 and V_mean > 55:
#                     label = "raw"
#                 elif H_YELLOW_MIN <= H_med < H_YELLOW_MAX and S_mean > 40 and V_mean > 60:
#                     label = "ripe"
#                 else:
#                     label = "rotten" if dark_frac >= 0.55 else "overripe"

#             if label == "unripe" and is_spotty_overripe and conf >= 0.50:
#                 label = "overripe"
#             # ============== END: HSV + spot detection block ==============

#             # สีของกรอบตาม label
#             color = (255, 255, 255)
#             if label == "raw":
#                 color = (0, 255, 255)
#             elif label == "unripe":
#                 color = (0, 165, 255)
#             elif label == "ripe":
#                 color = (0, 255, 0)
#             elif label == "overripe":
#                 color = (60, 20, 140)
#             elif label == "rotten":
#                 color = (0, 0, 255)

#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#             labels_info.append({
#                 "index": i + 1,
#                 "label": label,
#                 "confidence": round(conf, 2)
#             })

#         cv2.imwrite(output_path, img)
#         print(f"Output image saved to {output_path}")
#         return labels_info, output_path

#     except Exception as e:
#         print(f"Error during banana detection: {e}")
#         return [], os.path.join("static", "error_processing.jpg")


# #-------#
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os

# # ===== Load model =====
# model = None
# try:
#     model = YOLO("best.pt")
#     print("Fine-tuned YOLO model loaded successfully.")
# except Exception as e:
#     print(f"Error loading fine-tuned YOLO model (best.pt): {e}")
#     print("Please ensure 'best.pt' is in the same directory as this script.")


# def _run_yolo(img, conf=0.5, iou=0.45, imgsz=960, augment=False, verbose=False):
#     """ห่อฟังก์ชัน predict ของ YOLO ให้เรียกใช้ง่าย"""
#     return model.predict(img, conf=conf, iou=iou, imgsz=imgsz, augment=augment, verbose=verbose)


# def _roi_from_hsv(img):
#     """
#     หา ROI คร่าว ๆ ของกล้วยจากสีผิว (HSV) เพื่อลดพื้นหลังลวดลายจัด
#     คืนค่า: (x1,y1,x2,y2) ของ ROI ในพิกัดภาพเดิม หรือ None ถ้าหาไม่ได้
#     """
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

#     # mask พื้นผิวผลไม้ + กรองแสง/เงา
#     mask = ((S > 35) & (V > 35) & (V < 250)).astype(np.uint8) * 255

#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

#     cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return None

#     cnt = max(cnts, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(cnt)

#     # ขยายกรอบเผื่อ 10%
#     pad_x = int(w * 0.1)
#     pad_y = int(h * 0.1)
#     H_img, W_img = img.shape[:2]
#     x1 = max(0, x - pad_x)
#     y1 = max(0, y - pad_y)
#     x2 = min(W_img - 1, x + w + pad_x)
#     y2 = min(H_img - 1, y + h + pad_y)

#     # กันกรอบเล็กเกิน
#     if (x2 - x1) < 64 or (y2 - y1) < 64:
#         return None
#     return (x1, y1, x2, y2)


# def detect_banana(image_path):
#     """
#     ตรวจจับกล้วยและจำแนกระดับความสุก: raw, unripe, ripe, overripe, rotten
#     return: (labels_info: list[dict], output_path: str)
#     """
#     if model is None:
#         print("Error: YOLO model is not loaded. Cannot perform detection.")
#         return [], os.path.join("static", "error_processing.jpg")

#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not load image from {image_path}")
#         return [], os.path.join("static", "error_processing.jpg")

#     os.makedirs("static", exist_ok=True)
#     output_path = os.path.join("static", "result.jpg")

#     try:
#         # ========= PASS 1: ค่าปกติ =========
#         results_list = _run_yolo(img, conf=0.5, iou=0.45, imgsz=960, augment=False, verbose=False)
#         results = results_list[0]

#         # DEBUG: สรุปผลก่อน post-processing
#         if len(results.boxes) > 0:
#             for cls_id, cconf in zip(results.boxes.cls.cpu().numpy().astype(int),
#                                      results.boxes.conf.cpu().numpy()):
#                 print(f"YOLO predicted: {model.names[int(cls_id)]}  (conf={float(cconf):.2f})")

#         # ========= PASS 2: ไม่เจอ → ผ่อนปรน =========
#         if len(results.boxes) == 0:
#             print("[Two-Pass] Pass1 found 0 boxes. Retrying with lower conf & bigger imgsz...")
#             results_list = _run_yolo(img, conf=0.25, iou=0.50, imgsz=1280, augment=True, verbose=False)
#             results = results_list[0]

#         # ========= PASS 3 (ROI Fallback): ยังไม่เจอ → ครอป ROI แล้วลองอีก =========
#         roi_offset = (0, 0)
#         if len(results.boxes) == 0:
#             print("[Two-Pass] Pass2 still 0 boxes. Trying ROI fallback...")
#             roi = _roi_from_hsv(img)
#             if roi is not None:
#                 x1r, y1r, x2r, y2r = roi
#                 roi_img = img[y1r:y2r, x1r:x2r]
#                 roi_results_list = _run_yolo(roi_img, conf=0.25, iou=0.50, imgsz=1280, augment=True, verbose=False)
#                 roi_results = roi_results_list[0]
#                 if len(roi_results.boxes) > 0:
#                     results = roi_results
#                     roi_offset = (x1r, y1r)
#                     print(f"[ROI] Using ROI boxes, offset={roi_offset}")

#         labels_info = []

#         if len(results.boxes) == 0:
#             print("No bananas found after two-pass + ROI. Saving original image.")
#             cv2.imwrite(output_path, img)
#             return [{"index": 0, "label": "No banana detected", "confidence": 0.0}], output_path

#         H_img, W_img = img.shape[:2]

#         # ===== วาดผลลัพธ์แต่ละกล่อง =====
#         for i, box in enumerate(results.boxes.data):
#             # map พิกัดกลับถ้ามาจาก ROI
#             x1, y1, x2, y2 = box[:4]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             if roi_offset != (0, 0):
#                 x1 += roi_offset[0]; x2 += roi_offset[0]
#                 y1 += roi_offset[1]; y2 += roi_offset[1]

#             conf = float(box[4])
#             cls_id = int(box[5])

#             # clamp bbox
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(W_img - 1, x2), min(H_img - 1, y2)

#             # shrink 5% เพื่อตัดฉากหลัง
#             bw, bh = max(1, x2 - x1), max(1, y2 - y1)
#             padx, pady = max(1, int(0.05 * bw)), max(1, int(0.05 * bh))
#             xi1 = min(max(0, x1 + padx), W_img - 2)
#             yi1 = min(max(0, y1 + pady), H_img - 2)
#             xi2 = max(min(x2 - padx, W_img - 1), xi1 + 1)
#             yi2 = max(min(y2 - pady, H_img - 1), yi1 + 1)

#             crop = img[yi1:yi2, xi1:xi2]
#             if crop.size == 0:
#                 continue

#             # ---------- สี & texture ----------
#             hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#             H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

#             base_mask = np.where(((S > 35) & (V > 40) & (V < 250)), 255, 0).astype(np.uint8)
#             base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)
#             base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 2)
#             m = base_mask > 0
#             if not np.any(m):
#                 m = np.ones_like(base_mask, dtype=bool)

#             H_vals, S_vals, V_vals = H[m], S[m], V[m]
#             H_med  = float(np.median(H_vals)) if H_vals.size else 0.0
#             S_mean = float(np.mean(S_vals)) if S_vals.size else 0.0
#             V_mean = float(np.mean(V_vals)) if V_vals.size else 0.0

#             dark_frac      = float(np.mean(V_vals < 80)) if V_vals.size else 0.0
#             very_dark_frac = float(np.mean(V_vals < 45)) if V_vals.size else 0.0
#             lowsat_frac    = float(np.mean(S_vals < 60)) if S_vals.size else 0.0

#             # จุดสีน้ำตาล/ดำ
#             brown_mask = ((H >= 10) & (H <= 25) & (S >= 60) & (V <= 150) & m)
#             black_mask = ((V <= 45) & (S <= 140) & m)
#             spot_mask_raw = (brown_mask | black_mask).astype(np.uint8) * 255
#             spot_mask = cv2.morphologyEx(spot_mask_raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

#             num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats((spot_mask > 0).astype(np.uint8), connectivity=8)
#             areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
#             # >>> ปรับเกณฑ์ลด noise จุดเล็กมาก
#             spot_cnt  = int(((areas >= 80) & (areas <= 3000)).sum())
#             spot_area = int((spot_mask > 0).sum())
#             fg_area   = int(m.sum())
#             spot_frac = (spot_area / max(fg_area, 1))

#             # ขอบ/texture
#             edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150)
#             edge_density = float(np.mean((edges > 0) & m)) if edges.size else 0.0

#             # --- Green coverage & color cues (ช่วยทนไฟโทนอุ่น) ---
#             green_mask  = ((H >= 28) & (H <= 95) & m)    # รวมเขียวอมเหลือง
#             yellow_mask = ((H >= 18) & (H <= 35) & m)
#             green_ratio  = float(np.mean(green_mask))
#             yellow_ratio = float(np.mean(yellow_mask))

#             # ส่วนต่าง G-R (ภาพ BGR)
#             G = crop[:, :, 1][m]
#             R = crop[:, :, 2][m]
#             g_minus_r = float(np.mean(G.astype(np.float32) - R.astype(np.float32))) if G.size else 0.0

#             print(f"[DBG #{i+1}] pred={model.names[cls_id]} conf={conf:.2f}  "
#                   f"H_med={H_med:.1f} S={S_mean:.1f} V={V_mean:.1f}  "
#                   f"spot_frac={spot_frac:.3f} spot_cnt={spot_cnt}  "
#                   f"dark={dark_frac:.2f} vdark={very_dark_frac:.2f} edge={edge_density:.3f}  "
#                   f"green_ratio={green_ratio:.2f} g-r={g_minus_r:.1f}")

#             # ---- คำทำนายจากโมเดล (normalize) ----
#             pred_label_raw = model.names[cls_id]
#             pred_label = pred_label_raw.strip().lower()
#             label = pred_label

#             # ---- ช่วงสีโดยประมาณ (OpenCV H: 0–179)
#             H_YELLOW_MIN, H_YELLOW_MAX           = 20, 35
#             H_GREENYELLOW_MIN, H_GREENYELLOW_MAX = 35, 55
#             H_GREEN_MIN, H_GREEN_MAX             = 55, 85

#             # เดา label จาก "สี"
#             color_guess = None
#             if (H_GREEN_MIN <= H_med <= H_GREEN_MAX) and (S_mean > 45) and (V_mean > 55):
#                 color_guess = "raw"
#             elif (H_GREENYELLOW_MIN <= H_med < H_GREEN_MIN) and (S_mean > 40):
#                 color_guess = "unripe"
#             elif (H_YELLOW_MIN <= H_med <= H_YELLOW_MAX) and (S_mean > 35) and (V_mean > 65):
#                 color_guess = "ripe"

#             # สัญญาณหนักจริงของเน่า/งอม
#             strong_rotten = (
#                 (very_dark_frac >= 0.65 and H_med < 50) or
#                 ((dark_frac >= 0.75) and (lowsat_frac >= 0.60) and H_med < 50) or
#                 ((V_mean < 60) and (S_mean < 55) and (edge_density < 0.02) and H_med < 50)
#             )
#             strong_overripe = (
#                 (spot_frac >= 0.08) or
#                 (spot_cnt  >= 10)  or
#                 ((dark_frac >= 0.35) and (edge_density >= 0.02))
#             )

#             # ===== ตรรกะตัดสินใจ =====
#             if strong_rotten:
#                 label = "rotten"
#             else:
#                 trust_model = (conf >= 0.70)

#                 # ผ่อนเล็กน้อย ให้ทนฝุ่น/รอยเล็ก
#                 weak_rot_signals = (dark_frac < 0.25 and very_dark_frac < 0.08 and spot_frac < 0.07)

#                 # โมเดลว่า rotten/overripe แต่หลักฐานสีอ่อน → เชื่อสี
#                 if pred_label in ("rotten", "overripe") and weak_rot_signals and color_guess in ("raw", "unripe", "ripe"):
#                     label = color_guess
#                     print(f"[DBG #{i+1}] override rotten/overripe -> {label} (weak rot signals)")

#                 # ✅ เขียวชัดเจน → force override แม้ conf สูง
#                 elif pred_label in ("rotten", "overripe") \
#                      and (green_ratio >= 0.55 or g_minus_r > 8.0 or (28 <= H_med <= 55)) \
#                      and dark_frac < 0.25 and spot_frac < 0.08:
#                     label = "unripe" if H_med < 58 else "raw"
#                     print(f"[DBG #{i+1}] force-override {pred_label} -> {label} "
#                           f"(green_ratio={green_ratio:.2f}, g-r={g_minus_r:.1f}, H_med={H_med:.1f})")

#                 elif strong_overripe and conf < 0.60:
#                     label = "overripe"

#                 elif trust_model:
#                     label = pred_label

#                 elif color_guess is not None:
#                     label = color_guess
#                 # else คง pred_label

#             # Guard ท้าย: ใช้สัดส่วนเขียวกันพลาด
#             if label == "rotten" and green_ratio >= 0.55 and V_mean > 80 and dark_frac < 0.25:
#                 print(f"[DBG #{i+1}] final-guard: rotten -> unripe (green_ratio={green_ratio:.2f})")
#                 label = "unripe"
#             elif label in ("rotten", "overripe") and green_ratio >= 0.70 and V_mean > 80:
#                 print(f"[DBG #{i+1}] final-guard: {label} -> raw (very green, green_ratio={green_ratio:.2f})")
#                 label = "raw"

#             # สีกรอบตาม label
#             color = (255, 255, 255)
#             if label == "raw":
#                 color = (0, 255, 255)
#             elif label == "unripe":
#                 color = (0, 165, 255)
#             elif label == "ripe":
#                 color = (0, 255, 0)
#             elif label == "overripe":
#                 color = (60, 20, 140)
#             elif label == "rotten":
#                 color = (0, 0, 255)

#             # วาดกรอบ + ป้าย
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#             labels_info.append({"index": i + 1, "label": label, "confidence": round(conf, 2)})

#         cv2.imwrite(output_path, img)
#         print(f"Output image saved to {output_path}")
#         return labels_info, output_path

#     except Exception as e:
#         print(f"Error during banana detection: {e}")
#         return [], os.path.join("static", "error_processing.jpg")

# #---#


# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os

# # =========================
# # Load model
# # =========================
# model = None
# try:
#     model = YOLO("best.pt")
#     print("Fine-tuned YOLO model loaded successfully.")
# except Exception as e:
#     print(f"Error loading fine-tuned YOLO model (best.pt): {e}")
#     print("Please ensure 'best.pt' is in the same directory as this script.")


# def _run_yolo(img, conf=0.5, iou=0.45, imgsz=960, augment=False, verbose=False):
#     """Wrapper เรียก predict ของ YOLO"""
#     return model.predict(
#         img, conf=conf, iou=iou, imgsz=imgsz, augment=augment, verbose=verbose
#     )


# def _roi_from_hsv(img):
#     """
#     หา ROI คร่าว ๆ ของกล้วยจากสีผิว (HSV) เพื่อลดพื้นหลัง
#     คืนค่า: (x1,y1,x2,y2) หรือ None ถ้าหาไม่ได้
#     """
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

#     mask = ((S > 35) & (V > 35) & (V < 250)).astype(np.uint8) * 255
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

#     cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return None

#     cnt = max(cnts, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(cnt)

#     # เผื่อขอบ 10%
#     pad_x, pad_y = int(w * 0.1), int(h * 0.1)
#     H_img, W_img = img.shape[:2]
#     x1 = max(0, x - pad_x)
#     y1 = max(0, y - pad_y)
#     x2 = min(W_img - 1, x + w + pad_x)
#     y2 = min(H_img - 1, y + h + pad_y)

#     if (x2 - x1) < 64 or (y2 - y1) < 64:
#         return None
#     return (x1, y1, x2, y2)


# def detect_banana(image_path):
#     """
#     ตรวจจับกล้วยและจำแนกระดับความสุก: raw, unripe, ripe, overripe, rotten
#     return: (labels_info: list[dict], output_path: str)
#     """
#     if model is None:
#         print("Error: YOLO model is not loaded. Cannot perform detection.")
#         return [], os.path.join("static", "error_processing.jpg")

#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not load image from {image_path}")
#         return [], os.path.join("static", "error_processing.jpg")

#     os.makedirs("static", exist_ok=True)
#     output_path = os.path.join("static", "result.jpg")

#     try:
#         # ---------- PASS 1 ----------
#         results_list = _run_yolo(img, conf=0.5, iou=0.45, imgsz=960, augment=False, verbose=False)
#         results = results_list[0]

#         # ---------- PASS 2 ----------
#         if len(results.boxes) == 0:
#             print("[Two-Pass] Pass1 found 0 boxes. Retrying with lower conf & bigger imgsz...")
#             results_list = _run_yolo(img, conf=0.25, iou=0.50, imgsz=1280, augment=True, verbose=False)
#             results = results_list[0]

#         # ---------- PASS 3 (ROI Fallback) ----------
#         roi_offset = (0, 0)
#         if len(results.boxes) == 0:
#             print("[Two-Pass] Pass2 still 0 boxes. Trying ROI fallback...")
#             roi = _roi_from_hsv(img)
#             if roi is not None:
#                 x1r, y1r, x2r, y2r = roi
#                 roi_img = img[y1r:y2r, x1r:x2r]
#                 roi_results_list = _run_yolo(roi_img, conf=0.25, iou=0.50, imgsz=1280, augment=True, verbose=False)
#                 roi_results = roi_results_list[0]
#                 if len(roi_results.boxes) > 0:
#                     results = roi_results
#                     roi_offset = (x1r, y1r)
#                     print(f"[ROI] Using ROI boxes, offset={roi_offset}")

#         labels_info = []

#         if len(results.boxes) == 0:
#             print("No bananas found after two-pass + ROI. Saving original image.")
#             cv2.imwrite(output_path, img)
#             return [{"index": 0, "label": "No banana detected", "confidence": 0.0}], output_path

#         H_img, W_img = img.shape[:2]

#         # ---------- วนทุกกล่อง ----------
#         for i, box in enumerate(results.boxes.data):
#             x1, y1, x2, y2 = map(int, box[:4])
#             conf = float(box[4])
#             cls_id = int(box[5])

#             if roi_offset != (0, 0):
#                 x1 += roi_offset[0]; x2 += roi_offset[0]
#                 y1 += roi_offset[1]; y2 += roi_offset[1]

#             # clamp
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(W_img - 1, x2), min(H_img - 1, y2)

#             # เล็มขอบ 5% เพื่อตัดฉากหลัง
#             bw, bh = max(1, x2 - x1), max(1, y2 - y1)
#             padx, pady = max(1, int(0.05 * bw)), max(1, int(0.05 * bh))
#             xi1 = min(max(0, x1 + padx), W_img - 2)
#             yi1 = min(max(0, y1 + pady), H_img - 2)
#             xi2 = max(min(x2 - padx, W_img - 1), xi1 + 1)
#             yi2 = max(min(y2 - pady, H_img - 1), yi1 + 1)

#             crop = img[yi1:yi2, xi1:xi2]
#             if crop.size == 0:
#                 continue

#             # ---------- วิเคราะห์สี/พื้นผิว ----------
#             hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#             H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

#             base_mask = ((S > 35) & (V > 40) & (V < 250)).astype(np.uint8)
#             m = base_mask > 0
#             if not np.any(m):
#                 m = np.ones_like(base_mask, dtype=bool)

#             H_vals, S_vals, V_vals = H[m], S[m], V[m]
#             H_med  = float(np.median(H_vals)) if H_vals.size else 0.0
#             S_mean = float(np.mean(S_vals)) if S_vals.size else 0.0
#             V_mean = float(np.mean(V_vals)) if V_vals.size else 0.0

#             dark_frac      = float(np.mean(V_vals < 80)) if V_vals.size else 0.0
#             very_dark_frac = float(np.mean(V_vals < 45)) if V_vals.size else 0.0
#             lowsat_frac    = float(np.mean(S_vals < 60)) if S_vals.size else 0.0

#             edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150)
#             edge_density = float(np.mean((edges > 0) & m)) if edges.size else 0.0

#             # ---- ช่วงสีโดยประมาณ (OpenCV H: 0–179)
#             H_YELLOW_MIN, H_YELLOW_MAX           = 20, 32
#             H_GREENYELLOW_MIN, H_GREENYELLOW_MAX = 32, 55
#             H_GREEN_MIN, H_GREEN_MAX             = 55, 85

#             # --- Green coverage (อย่านับเหลืองเป็นเขียว)
#             green_mask  = ((H >= 38) & (H <= 85) & m)   # ตัดเหลืองออกชัด
#             yellow_mask = ((H >= 20) & (H <= 32) & m)
#             green_ratio  = float(np.mean(green_mask))
#             yellow_ratio = float(np.mean(yellow_mask))

#             # สัญญาณเน่า/งอม
#             strong_rotten = (
#                 (very_dark_frac >= 0.65 and H_med < 50)
#                 or ((dark_frac >= 0.75) and (lowsat_frac >= 0.60) and H_med < 50)
#                 or ((V_mean < 60) and (S_mean < 55) and (edge_density < 0.02) and H_med < 50)
#             )
#             strong_overripe = (dark_frac >= 0.35 and edge_density >= 0.02)

#             # คำทำนายจากโมเดล
#             pred_label_raw = model.names[cls_id]
#             pred_label = pred_label_raw.strip().lower()
#             label = pred_label

#             print(f"[DBG #{i+1}] pred={pred_label} conf={conf:.2f}  "
#                   f"H_med={H_med:.1f} S={S_mean:.1f} V={V_mean:.1f}  "
#                   f"dark={dark_frac:.2f} vdark={very_dark_frac:.2f} edge={edge_density:.3f}  "
#                   f"green_ratio={green_ratio:.2f} yellow_ratio={yellow_ratio:.2f}")

#             # ===== ตรรกะตัดสินใจหลัก =====
#             if strong_rotten:
#                 label = "rotten"
#             else:
#                 trust_model = (conf >= 0.70)

#                 # เชื่อโมเดลถ้าเดา raw/unripe และไม่เจอสัญญาณงอมหนัก
#                 if pred_label in ("raw", "unripe") and conf >= 0.55 and not strong_overripe:
#                     label = pred_label
#                 else:
#                     # ถ้าโมเดลว่า rotten/overripe แต่สัญญาณมืด/เหลืองน้อยมาก → เอียงไปฝั่งเขียว
#                     weak_rot_signals = (dark_frac < 0.25 and very_dark_frac < 0.08 and yellow_ratio < 0.07)

#                     if pred_label in ("rotten", "overripe") and weak_rot_signals and \
#                        (green_ratio >= 0.55 or (H_GREENYELLOW_MIN <= H_med <= H_GREENYELLOW_MAX)):
#                         label = "unripe" if H_med < 58 else "raw"

#                     elif strong_overripe and conf < 0.60:
#                         label = "overripe"

#                     elif trust_model:
#                         label = pred_label
#                     # else: คง label เดิม

#             # --- Final refinement: ชั่งน้ำหนัก raw vs unripe จากเขียว/เหลืองจริงบนผิว ---
#             if label in ("raw", "unripe"):
#                 strong_green = (green_ratio >= 0.75 and yellow_ratio < 0.04 and H_med >= H_GREEN_MIN)
#                 has_visible_yellow = (yellow_ratio >= 0.06) or (H_GREENYELLOW_MIN <= H_med <= H_GREENYELLOW_MAX)

#                 if strong_green:
#                     label = "raw"
#                 elif has_visible_yellow:
#                     label = "unripe"

#                 print(f"[REFINE #{i+1}] raw/unripe -> {label} "
#                       f"(H_med={H_med:.1f}, green={green_ratio:.2f}, yellow={yellow_ratio:.2f})")

#             # ---------- วาดผล ----------
#             color = (255, 255, 255)
#             if label == "raw":
#                 color = (0, 255, 255)
#             elif label == "unripe":
#                 color = (0, 165, 255)
#             elif label == "ripe":
#                 color = (0, 255, 0)
#             elif label == "overripe":
#                 color = (60, 20, 140)
#             elif label == "rotten":
#                 color = (0, 0, 255)

#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#             labels_info.append({"index": i + 1, "label": label, "confidence": round(conf, 2)})

#         cv2.imwrite(output_path, img)
#         print(f"Output image saved to {output_path}")
#         return labels_info, output_path

#     except Exception as e:
#         print(f"Error during banana detection: {e}")
#         return [], os.path.join("static", "error_processing.jpg")

#--------#
# # -*- coding: utf-8 -*-
# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO

# # =========================
# # Tunables (ปรับได้ตามต้องการ)
# # =========================
# FORCE_ROTTEN_WHITE_FRAC = 0.02
# FORCE_ROTTEN_PULP_FRAC  = 0.10
# ROTTEN_BLACK_FRAC       = 0.25
# OVERRIPE_YELLOW_RATIO   = 0.10
# OVERRIPE_MAX_DARK       = 0.40
# OVERRIPE_MAX_VDARK      = 0.20

# # ตัดพื้นหลังหลายสี (OpenCV HSV)
# EXCLUDE_COLOR_BANDS = [
#     (95, 130, 45, 60),    # blue
#     (85,  95, 45, 60),    # cyan-ish
#     (130, 170, 40, 50),   # purple/magenta
#     (0,   10, 45, 60),    # red (low end)
#     (170, 179, 45, 60),   # red (upper end)
# ]

# # =========================
# # Load model
# # =========================
# model = None
# try:
#     model = YOLO("best.pt")
#     print("Fine-tuned YOLO model loaded successfully.")
# except Exception as e:
#     print(f"Error loading fine-tuned YOLO model (best.pt): {e}")
#     print("Please ensure 'best.pt' is in the same directory as this script.")

# # =========================
# # Helpers
# # =========================
# def _run_yolo(img, conf=0.5, iou=0.45, imgsz=960, augment=False, verbose=False):
#     return model.predict(img, conf=conf, iou=iou, imgsz=imgsz, augment=augment, verbose=verbose)

# def _make_banana_mask(hsv, tighten=False):
#     H = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]

#     yellow_band = (H >= 20) & (H <= 35)
#     green_band  = (H >= 38) & (H <= 95)
#     brown_band  = (H >= 10) & (H <= 25) & (S >= 40) & (V <= 180)
#     dark_band   = (V <= 60) & (S <= 120)

#     banana = (yellow_band | green_band | brown_band | dark_band)
#     banana &= (S > 25) & (V > 35)

#     remove = np.zeros_like(banana, dtype=bool)
#     for (hmin, hmax, smin, vmin) in EXCLUDE_COLOR_BANDS:
#         remove |= ((H >= hmin) & (H <= hmax) & (S >= smin) & (V >= vmin))
#     banana_h_buffer = ((H >= 16) & (H <= 38)) | ((H >= 55) & (H <= 95))
#     remove &= ~banana_h_buffer

#     banana = banana.astype(np.uint8) * 255
#     remove = remove.astype(np.uint8) * 255

#     if tighten:
#         banana = cv2.morphologyEx(banana, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8), 1)
#         banana = cv2.morphologyEx(banana, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), 2)
#         remove = cv2.morphologyEx(remove, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8), 1)
#     else:
#         banana = cv2.morphologyEx(banana, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8), 2)
#         banana = cv2.morphologyEx(banana, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8), 1)
#         remove = cv2.morphologyEx(remove, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8), 1)

#     mask = cv2.bitwise_and(banana, cv2.bitwise_not(remove))
#     return (mask > 0)

# def _roi_from_hsv(img):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask = _make_banana_mask(hsv, tighten=False).astype(np.uint8) * 255
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

#     cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return None

#     cnt = max(cnts, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(cnt)

#     pad_x, pad_y = int(w * 0.1), int(h * 0.1)
#     H_img, W_img = img.shape[:2]
#     x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
#     x2 = min(W_img - 1, x + w + pad_x); y2 = min(H_img - 1, y + h + pad_y)

#     if (x2 - x1) < 64 or (y2 - y1) < 64:
#         return None
#     return (x1, y1, x2, y2)

# def _nms_boxes(boxes, scores, iou_thr=0.35):
#     if len(boxes) == 0:
#         return boxes, scores
#     x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
#     areas = (x2-x1+1) * (y2-y1+1)
#     order = scores.argsort()[::-1]
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#         w = np.maximum(0.0, xx2-xx1+1)
#         h = np.maximum(0.0, yy2-yy1+1)
#         inter = w*h
#         iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
#         inds = np.where(iou <= iou_thr)[0]
#         order = order[inds + 1]
#     return boxes[keep], scores[keep]

# def _propose_banana_boxes(img):
#     H_img, W_img = img.shape[:2]
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     banana_mask = _make_banana_mask(hsv, tighten=False).astype(np.uint8) * 255

#     cnts, _ = cv2.findContours(banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return []

#     props = []
#     img_area = H_img * W_img
#     for c in cnts:
#         x,y,w,h = cv2.boundingRect(c)
#         area = w*h
#         if area < 0.01*img_area:
#             continue
#         ar = max(w,h)/max(1.0, min(w,h))
#         if ar < 1.6:   # กล้วยมักยาว
#             continue
#         score = (area/img_area) * ar
#         props.append(((x,y,x+w,y+h), score))

#     if not props:
#         return []
#     boxes, scores = zip(*props)
#     boxes = np.array(boxes, np.float32)
#     scores = np.array(scores, np.float32)
#     boxes, scores = _nms_boxes(boxes, scores, iou_thr=0.35)
#     keep = min(3, boxes.shape[0])
#     return [tuple(map(int, boxes[i])) + (float(scores[i]),) for i in range(keep)]

# # ---- YOLO result shim (แก้บั๊ก len(Dummy)) ----
# class _BoxesShim:
#     def __init__(self, data: np.ndarray):
#         self.data = data  # shape (N,6) = [x1,y1,x2,y2,conf,cls]
#     def __len__(self):
#         return 0 if self.data is None else int(self.data.shape[0])

# class _ResultShim:
#     def __init__(self, data: np.ndarray):
#         self.boxes = _BoxesShim(data)

# # =========================
# # Main
# # =========================
# def detect_banana(image_path):
#     if model is None:
#         print("Error: YOLO model is not loaded. Cannot perform detection.")
#         return [], os.path.join("static", "error_processing.jpg")

#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not load image from {image_path}")
#         return [], os.path.join("static", "error_processing.jpg")

#     os.makedirs("static", exist_ok=True)
#     output_path = os.path.join("static", "result.jpg")

#     try:
#         # ---------- PASS 1 ----------
#         results_list = _run_yolo(img, conf=0.5, iou=0.45, imgsz=960, augment=False, verbose=False)
#         results = results_list[0]

#         # ---------- PASS 2 ----------
#         if len(results.boxes) == 0:
#             print("[Two-Pass] Pass1 found 0 boxes. Retrying with lower conf & bigger imgsz...")
#             results_list = _run_yolo(img, conf=0.25, iou=0.50, imgsz=1280, augment=True, verbose=False)
#             results = results_list[0]

#         # ---------- PASS 3 (ROI) ----------
#         roi_offset = (0, 0)
#         if len(results.boxes) == 0:
#             print("[Two-Pass] Pass2 still 0 boxes. Trying ROI fallback...")
#             roi = _roi_from_hsv(img)
#             if roi is not None:
#                 x1r, y1r, x2r, y2r = roi
#                 roi_img = img[y1r:y2r, x1r:x2r]
#                 roi_results_list = _run_yolo(roi_img, conf=0.25, iou=0.50, imgsz=1280, augment=True, verbose=False)
#                 roi_results = roi_results_list[0]
#                 if len(roi_results.boxes) > 0:
#                     results = roi_results
#                     roi_offset = (x1r, y1r)
#                     print(f"[ROI] Using ROI boxes, offset={roi_offset}")

#         # ---------- PASS 4 (Shape proposer) ----------
#         if len(results.boxes) == 0:
#             print("[Proposer] YOLO still 0 boxes. Using color+shape proposer...")
#             props = _propose_banana_boxes(img)
#             if props:
#                 rows = []
#                 for (x1, y1, x2, y2, s) in props:
#                     rows.append([x1, y1, x2, y2, 0.30, 2.0])  # conf=0.30, cls=2 ('ripe' placeholder)
#                 data = np.array(rows, dtype=np.float32)
#                 results = _ResultShim(data)  # <<< FIX: มี __len__ แล้ว
#             else:
#                 print("No proposals generated.")

#         labels_info = []
#         if len(results.boxes) == 0:
#             print("No bananas found after two-pass + ROI + proposer.")
#             # กันภาพ error: เซฟภาพเดิมเป็น result.jpg เลย
#             cv2.imwrite(output_path, img)
#             return [{"index": 0, "label": "No banana detected", "confidence": 0.0}], output_path

#         H_img, W_img = img.shape[:2]

#         # ---------- วนทุกกล่อง ----------
#         for i, box in enumerate(results.boxes.data):
#             x1, y1, x2, y2 = map(int, box[:4])
#             conf = float(box[4])
#             cls_id = int(box[5]) if box.shape[0] >= 6 else 2

#             if roi_offset != (0, 0):
#                 x1 += roi_offset[0]; x2 += roi_offset[0]
#                 y1 += roi_offset[1]; y2 += roi_offset[1]

#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(W_img - 1, x2), min(H_img - 1, y2)

#             bw, bh = max(1, x2 - x1), max(1, y2 - y1)
#             padx, pady = max(1, int(0.05 * bw)), max(1, int(0.05 * bh))
#             xi1 = min(max(0, x1 + padx), W_img - 2)
#             yi1 = min(max(0, y1 + pady), H_img - 2)
#             xi2 = max(min(x2 - padx, W_img - 1), xi1 + 1)
#             yi2 = max(min(y2 - pady, H_img - 1), yi1 + 1)

#             crop = img[yi1:yi2, xi1:xi2]
#             if crop.size == 0:
#                 continue

#             hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#             m = _make_banana_mask(hsv, tighten=True)
#             Hc, Sc, Vc = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
#             if not np.any(m):
#                 m = ((Sc > 35) & (Vc > 40) & (Vc < 250))

#             H_vals, S_vals, V_vals = Hc[m], Sc[m], Vc[m]
#             H_med  = float(np.median(H_vals)) if H_vals.size else 0.0
#             S_mean = float(np.mean(S_vals)) if S_vals.size else 0.0
#             V_mean = float(np.mean(V_vals)) if V_vals.size else 0.0

#             dark_frac      = float(np.mean(V_vals < 80)) if V_vals.size else 0.0
#             very_dark_frac = float(np.mean(V_vals < 45)) if V_vals.size else 0.0
#             lowsat_frac    = float(np.mean(S_vals < 60)) if S_vals.size else 0.0

#             edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150)
#             edge_density = float(np.mean((edges > 0) & m)) if edges.size else 0.0

#             H_YELLOW_MIN, H_YELLOW_MAX           = 20, 32
#             H_GREENYELLOW_MIN, H_GREENYELLOW_MAX = 32, 55
#             H_GREEN_MIN, H_GREEN_MAX             = 55, 85

#             green_mask  = ((Hc >= 38) & (Hc <= 85) & m)
#             yellow_mask = ((Hc >= 20) & (Hc <= 32) & m)
#             green_ratio  = float(np.mean(green_mask))
#             yellow_ratio = float(np.mean(yellow_mask))

#             white_frac = float(np.mean(((Sc < 35) & (Vc > 200) & m)))
#             pulp_frac  = float(np.mean(((Sc < 50) & (Vc > 130) & m)))
#             black_frac = float(np.mean(((Vc <= 45) & m)))
#             brown_frac = float(np.mean(((Hc >= 10) & (Hc <= 25) & (Sc >= 60) & (Vc <= 160) & m)))

#             cracked_or_peeled = (white_frac >= FORCE_ROTTEN_WHITE_FRAC) or (pulp_frac >= FORCE_ROTTEN_PULP_FRAC)
#             yellowish_enough  = ((H_YELLOW_MIN <= H_med <= H_YELLOW_MAX) or (yellow_ratio >= OVERRIPE_YELLOW_RATIO))

#             pred_label_raw = model.names[cls_id] if hasattr(model, "names") else "ripe"
#             pred_label = str(pred_label_raw).strip().lower()
#             label = pred_label

#             # Override แตก/ปอก/เยิ้ม
#             if cracked_or_peeled and (black_frac >= ROTTEN_BLACK_FRAC or very_dark_frac >= OVERRIPE_MAX_VDARK or dark_frac >= 0.55):
#                 label = "rotten"
#             elif cracked_or_peeled and yellowish_enough and (dark_frac < OVERRIPE_MAX_DARK) and (very_dark_frac < OVERRIPE_MAX_VDARK):
#                 label = "overripe"
#             else:
#                 strong_rotten = (
#                     (very_dark_frac >= 0.65 and H_med < 50)
#                     or ((dark_frac >= 0.75) and (lowsat_frac >= 0.60) and H_med < 50)
#                     or ((V_mean < 60) and (S_mean < 55) and (edge_density < 0.02) and H_med < 50)
#                 )
#                 strong_overripe = (dark_frac >= 0.35 and edge_density >= 0.02)

#                 if strong_rotten:
#                     label = "rotten"
#                 else:
#                     trust_model = (conf >= 0.70)
#                     if pred_label in ("raw", "unripe") and conf >= 0.55 and not strong_overripe:
#                         label = pred_label
#                     else:
#                         weak_rot_signals = (dark_frac < 0.25 and very_dark_frac < 0.08 and yellow_ratio < 0.07)
#                         if pred_label in ("rotten", "overripe") and weak_rot_signals and \
#                            (green_ratio >= 0.55 or (H_GREENYELLOW_MIN <= H_med <= H_GREENYELLOW_MAX)):
#                             label = "unripe" if H_med < 58 else "raw"
#                         elif strong_overripe and conf < 0.60:
#                             label = "overripe"
#                         elif trust_model:
#                             label = pred_label

#                 if label in ("raw", "unripe"):
#                     strong_green = (green_ratio >= 0.75 and yellow_ratio < 0.04 and H_med >= H_GREEN_MIN)
#                     has_visible_yellow = (yellow_ratio >= 0.06) or (H_GREENYELLOW_MIN <= H_med <= H_GREENYELLOW_MAX)
#                     if strong_green:
#                         label = "raw"
#                     elif has_visible_yellow:
#                         label = "unripe"

#             # วาดผล
#             color = (255, 255, 255)
#             if label == "raw": color = (0, 255, 255)
#             elif label == "unripe": color = (0, 165, 255)
#             elif label == "ripe": color = (0, 255, 0)
#             elif label == "overripe": color = (60, 20, 140)
#             elif label == "rotten": color = (0, 0, 255)

#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#             labels_info.append({"index": i + 1, "label": label, "confidence": round(conf, 2)})

#         cv2.imwrite(output_path, img)
#         print(f"Output image saved to {output_path}")
#         return labels_info, output_path

#     except Exception as e:
#         print(f"Error during banana detection: {e}")
#         # กันภาพ error ไม่เจอไฟล์: เซฟภาพเดิมเป็น error_processing.jpg
#         err_path = os.path.join("static", "error_processing.jpg")
#         try:
#             cv2.imwrite(err_path, img)
#         except:
#             pass
#         return [], err_path


#-----#
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os

# # =========================
# # Load model
# # =========================
# model = None
# try:
#     model = YOLO("best.pt")
#     print("Fine-tuned YOLO model loaded successfully.")
# except Exception as e:
#     print(f"Error loading fine-tuned YOLO model (best.pt): {e}")
#     print("Please ensure 'best.pt' is in the same directory as this script.")


# def _run_yolo(img, conf=0.5, iou=0.45, imgsz=960, augment=False, verbose=False):
#     """Wrapper เรียก predict ของ YOLO"""
#     return model.predict(
#         img, conf=conf, iou=iou, imgsz=imgsz, augment=augment, verbose=verbose
#     )


# def _roi_from_hsv(img):
#     """
#     หา ROI คร่าว ๆ ของกล้วยจากสีผิว (HSV) เพื่อลดพื้นหลัง
#     คืนค่า: (x1,y1,x2,y2) หรือ None ถ้าหาไม่ได้
#     """
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

#     mask = ((S > 35) & (V > 35) & (V < 250)).astype(np.uint8) * 255
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

#     cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return None

#     cnt = max(cnts, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(cnt)

#     # เผื่อขอบ 10%
#     pad_x, pad_y = int(w * 0.1), int(h * 0.1)
#     H_img, W_img = img.shape[:2]
#     x1 = max(0, x - pad_x)
#     y1 = max(0, y - pad_y)
#     x2 = min(W_img - 1, x + w + pad_x)
#     y2 = min(H_img - 1, y + h + pad_y)

#     if (x2 - x1) < 64 or (y2 - y1) < 64:
#         return None
#     return (x1, y1, x2, y2)


# def detect_banana(image_path):
#     """
#     ตรวจจับกล้วยและจำแนกระดับความสุก: raw, unripe, ripe, overripe, rotten
#     return: (labels_info: list[dict], output_path: str)
#     """
#     if model is None:
#         print("Error: YOLO model is not loaded. Cannot perform detection.")
#         return [], os.path.join("static", "error_processing.jpg")

#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not load image from {image_path}")
#         return [], os.path.join("static", "error_processing.jpg")

#     os.makedirs("static", exist_ok=True)
#     output_path = os.path.join("static", "result.jpg")

#     try:
#         # ---------- PASS 1 ----------
#         results_list = _run_yolo(img, conf=0.5, iou=0.45, imgsz=960, augment=False, verbose=False)
#         results = results_list[0]

#         # ---------- PASS 2 ----------
#         if len(results.boxes) == 0:
#             print("[Two-Pass] Pass1 found 0 boxes. Retrying with lower conf & bigger imgsz...")
#             results_list = _run_yolo(img, conf=0.25, iou=0.50, imgsz=1280, augment=True, verbose=False)
#             results = results_list[0]

#         # ---------- PASS 3 (ROI Fallback) ----------
#         roi_offset = (0, 0)
#         if len(results.boxes) == 0:
#             print("[Two-Pass] Pass2 still 0 boxes. Trying ROI fallback...")
#             roi = _roi_from_hsv(img)
#             if roi is not None:
#                 x1r, y1r, x2r, y2r = roi
#                 roi_img = img[y1r:y2r, x1r:x2r]
#                 roi_results_list = _run_yolo(roi_img, conf=0.25, iou=0.50, imgsz=1280, augment=True, verbose=False)
#                 roi_results = roi_results_list[0]
#                 if len(roi_results.boxes) > 0:
#                     results = roi_results
#                     roi_offset = (x1r, y1r)
#                     print(f"[ROI] Using ROI boxes, offset={roi_offset}")

#         labels_info = []

#         if len(results.boxes) == 0:
#             print("No bananas found after two-pass + ROI. Saving original image.")
#             cv2.imwrite(output_path, img)
#             return [{"index": 0, "label": "No banana detected", "confidence": 0.0}], output_path

#         H_img, W_img = img.shape[:2]

#         # ---------- วนทุกกล่อง ----------
#         for i, box in enumerate(results.boxes.data):
#             x1, y1, x2, y2 = map(int, box[:4])
#             conf = float(box[4])
#             cls_id = int(box[5])

#             if roi_offset != (0, 0):
#                 x1 += roi_offset[0]; x2 += roi_offset[0]
#                 y1 += roi_offset[1]; y2 += roi_offset[1]

#             # clamp
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(W_img - 1, x2), min(H_img - 1, y2)

#             # เล็มขอบ 5% เพื่อตัดฉากหลัง
#             bw, bh = max(1, x2 - x1), max(1, y2 - y1)
#             padx, pady = max(1, int(0.05 * bw)), max(1, int(0.05 * bh))
#             xi1 = min(max(0, x1 + padx), W_img - 2)
#             yi1 = min(max(0, y1 + pady), H_img - 2)
#             xi2 = max(min(x2 - padx, W_img - 1), xi1 + 1)
#             yi2 = max(min(y2 - pady, H_img - 1), yi1 + 1)

#             crop = img[yi1:yi2, xi1:xi2]
#             if crop.size == 0:
#                 continue

#             # ---------- วิเคราะห์สี/พื้นผิว ----------
#             hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#             H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

#             base_mask = ((S > 35) & (V > 40) & (V < 250)).astype(np.uint8)
#             m = base_mask > 0
#             if not np.any(m):
#                 m = np.ones_like(base_mask, dtype=bool)

#             H_vals, S_vals, V_vals = H[m], S[m], V[m]
#             H_med  = float(np.median(H_vals)) if H_vals.size else 0.0
#             S_mean = float(np.mean(S_vals)) if S_vals.size else 0.0
#             V_mean = float(np.mean(V_vals)) if V_vals.size else 0.0

#             dark_frac      = float(np.mean(V_vals < 80)) if V_vals.size else 0.0
#             very_dark_frac = float(np.mean(V_vals < 45)) if V_vals.size else 0.0
#             lowsat_frac    = float(np.mean(S_vals < 60)) if S_vals.size else 0.0

#             edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150)
#             edge_density = float(np.mean((edges > 0) & m)) if edges.size else 0.0

#             # ---- ช่วงสีโดยประมาณ (OpenCV H: 0–179)
#             H_YELLOW_MIN, H_YELLOW_MAX           = 20, 32
#             H_GREENYELLOW_MIN, H_GREENYELLOW_MAX = 32, 55
#             H_GREEN_MIN, H_GREEN_MAX             = 55, 85

#             # --- Green coverage (อย่านับเหลืองเป็นเขียว)
#             green_mask  = ((H >= 38) & (H <= 85) & m)   # ตัดเหลืองออกชัด
#             yellow_mask = ((H >= 20) & (H <= 32) & m)
#             green_ratio  = float(np.mean(green_mask))
#             yellow_ratio = float(np.mean(yellow_mask))

#             # สัญญาณเน่า/งอม
#             strong_rotten = (
#                 (very_dark_frac >= 0.65 and H_med < 50)
#                 or ((dark_frac >= 0.75) and (lowsat_frac >= 0.60) and H_med < 50)
#                 or ((V_mean < 60) and (S_mean < 55) and (edge_density < 0.02) and H_med < 50)
#             )
#             strong_overripe = (dark_frac >= 0.35 and edge_density >= 0.02)

#             # คำทำนายจากโมเดล
#             pred_label_raw = model.names[cls_id]
#             pred_label = pred_label_raw.strip().lower()
#             label = pred_label

#             print(f"[DBG #{i+1}] pred={pred_label} conf={conf:.2f}  "
#                   f"H_med={H_med:.1f} S={S_mean:.1f} V={V_mean:.1f}  "
#                   f"dark={dark_frac:.2f} vdark={very_dark_frac:.2f} edge={edge_density:.3f}  "
#                   f"green_ratio={green_ratio:.2f} yellow_ratio={yellow_ratio:.2f}")

#             # ===== ตรรกะตัดสินใจหลัก =====
#             if strong_rotten:
#                 label = "rotten"
#             else:
#                 trust_model = (conf >= 0.70)

#                 # เชื่อโมเดลถ้าเดา raw/unripe และไม่เจอสัญญาณงอมหนัก
#                 if pred_label in ("raw", "unripe") and conf >= 0.55 and not strong_overripe:
#                     label = pred_label
#                 else:
#                     # ถ้าโมเดลว่า rotten/overripe แต่สัญญาณมืด/เหลืองน้อยมาก → เอียงไปฝั่งเขียว
#                     weak_rot_signals = (dark_frac < 0.25 and very_dark_frac < 0.08 and yellow_ratio < 0.07)

#                     if pred_label in ("rotten", "overripe") and weak_rot_signals and \
#                        (green_ratio >= 0.55 or (H_GREENYELLOW_MIN <= H_med <= H_GREENYELLOW_MAX)):
#                         label = "unripe" if H_med < 58 else "raw"

#                     elif strong_overripe and conf < 0.60:
#                         label = "overripe"

#                     elif trust_model:
#                         label = pred_label
#                     # else: คง label เดิม

#             # --- Final refinement: ชั่งน้ำหนัก raw vs unripe จากเขียว/เหลืองจริงบนผิว ---
#             if label in ("raw", "unripe"):
#                 strong_green = (green_ratio >= 0.75 and yellow_ratio < 0.04 and H_med >= H_GREEN_MIN)
#                 has_visible_yellow = (yellow_ratio >= 0.06) or (H_GREENYELLOW_MIN <= H_med <= H_GREENYELLOW_MAX)

#                 if strong_green:
#                     label = "raw"
#                 elif has_visible_yellow:
#                     label = "unripe"

#                 print(f"[REFINE #{i+1}] raw/unripe -> {label} "
#                       f"(H_med={H_med:.1f}, green={green_ratio:.2f}, yellow={yellow_ratio:.2f})")

#             # ---------- วาดผล ----------
#             color = (255, 255, 255)
#             if label == "raw":
#                 color = (0, 255, 255)
#             elif label == "unripe":
#                 color = (0, 165, 255)
#             elif label == "ripe":
#                 color = (0, 255, 0)
#             elif label == "overripe":
#                 color = (60, 20, 140)
#             elif label == "rotten":
#                 color = (0, 0, 255)

#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#             labels_info.append({"index": i + 1, "label": label, "confidence": round(conf, 2)})

#         cv2.imwrite(output_path, img)
#         print(f"Output image saved to {output_path}")
#         return labels_info, output_path

#     except Exception as e:
#         print(f"Error during banana detection: {e}")
#         return [], os.path.join("static", "error_processing.jpg")

#------#
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os

# # =========================
# # Load model
# # =========================
# model = None
# try:
#     model = YOLO("best.pt")
#     print("Fine-tuned YOLO model loaded successfully.")
# except Exception as e:
#     print(f"Error loading fine-tuned YOLO model (best.pt): {e}")
#     print("Please ensure 'best.pt' is in the same directory as this script.")


# def _run_yolo(img, conf=0.5, iou=0.45, imgsz=960, augment=False, verbose=False):
#     """Wrapper เรียก predict ของ YOLO"""
#     return model.predict(img, conf=conf, iou=iou, imgsz=imgsz, augment=augment, verbose=verbose)


# # ---------- Busy-background suppressor (ทั่วไป) ----------
# def _dominant_bg_hue_mask(hsv, keep_hue_ranges=((20, 95),), bins=180, topk=2, tol=5):
#     """
#     หา dominant hue ของทั้งภาพ (พื้นหลังกวน) โดย *ยกเว้น* ช่วงที่เราต้องการเก็บไว้ (20–95 = เหลือง-เขียว)
#     คืนค่า mask ของ 'พื้นหลังกวน' (True = ให้ตัดทิ้งตอนเก็บสถิติ)
#     """
#     H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

#     # พื้นที่ที่สีพอมีความหมาย
#     valid = (S > 30) & (V > 40)
#     if not np.any(valid):
#         return np.zeros_like(H, dtype=bool)

#     hvals = H[valid].ravel()

#     # ทำ histogram
#     hist, _ = np.histogram(hvals, bins=bins, range=(0,180))
#     # mask ช่องที่เป็นช่วงกล้วย (ไม่ให้เป็นผู้ต้องสงสัย)
#     keep = np.zeros_like(hist, dtype=bool)
#     for a,b in keep_hue_ranges:
#         keep[a:b+1] = True
#     hist_fg = hist.copy()
#     hist_fg[keep] = 0

#     # เลือก top-k โทนเด่นที่ *น่าจะเป็นพื้นหลัง*
#     top_idx = hist_fg.argsort()[::-1][:topk]
#     bg_mask = np.zeros_like(H, dtype=bool)
#     for idx in top_idx:
#         if hist_fg[idx] == 0: 
#             continue
#         hmin = max(0, idx - tol)
#         hmax = min(179, idx + tol)
#         bg_mask |= ((H >= hmin) & (H <= hmax))

#     return bg_mask


# def _roi_from_hsv(img):
#     """หา ROI คร่าว ๆ ของกล้วยจากสีผิว เพื่อช่วยตอน YOLO ไม่เจอ"""
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

#     # โทนกล้วยกว้าง ๆ
#     banana_mask = ((S > 35) & (V > 40) & (V < 250) & (H >= 18) & (H <= 95))

#     # ตัดพื้นหลังกวนที่เด่นมาก (ฮิสโตแกรม hue)
#     bg_dom = _dominant_bg_hue_mask(hsv, keep_hue_ranges=((18,95),), bins=180, topk=2, tol=6)
#     banana_mask = banana_mask & (~bg_dom)

#     kernel = np.ones((7,7), np.uint8)
#     banana_mask = cv2.morphologyEx(banana_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, kernel, 2)
#     banana_mask = cv2.morphologyEx(banana_mask, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8), 1)

#     cnts, _ = cv2.findContours(banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return None
#     cnt = max(cnts, key=cv2.contourArea)
#     x,y,w,h = cv2.boundingRect(cnt)

#     # เผื่อขอบ 10%
#     pad_x, pad_y = int(w*0.10), int(h*0.10)
#     H_img, W_img = img.shape[:2]
#     x1 = max(0, x - pad_x)
#     y1 = max(0, y - pad_y)
#     x2 = min(W_img-1, x + w + pad_x)
#     y2 = min(H_img-1, y + h + pad_y)

#     if (x2-x1) < 64 or (y2-y1) < 64:
#         return None
#     return (x1,y1,x2,y2)


# def detect_banana(image_path):
#     """
#     ตรวจจับกล้วยและจำแนกระดับความสุก: raw, unripe, ripe, overripe, rotten
#     return: (labels_info: list[dict], output_path: str)
#     """
#     if model is None:
#         print("Error: YOLO model is not loaded. Cannot perform detection.")
#         return [], os.path.join("static", "error_processing.jpg")

#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not load image from {image_path}")
#         return [], os.path.join("static", "error_processing.jpg")

#     os.makedirs("static", exist_ok=True)
#     output_path = os.path.join("static", "result.jpg")

#     try:
#         # ---------- PASS 1 ----------
#         results_list = _run_yolo(img, conf=0.5, iou=0.45, imgsz=960, augment=False, verbose=False)
#         results = results_list[0]

#         # ---------- PASS 2 ----------
#         if len(results.boxes) == 0:
#             print("[Two-Pass] Pass1 found 0 boxes. Retrying with lower conf & bigger imgsz...")
#             results_list = _run_yolo(img, conf=0.25, iou=0.50, imgsz=1280, augment=True, verbose=False)
#             results = results_list[0]

#         # ---------- PASS 3: ROI / Proposer ----------
#         roi_offset = (0,0)
#         need_proposer = False
#         if len(results.boxes) == 0:
#             need_proposer = True
#         else:
#             # ตรวจว่ากล่องที่เจอมี "ผิวกล้วยจริง" พอหรือไม่
#             x1,y1,x2,y2 = map(int, results.boxes.data[0][:4])
#             crop = img[max(0,y1):y2, max(0,x1):x2]
#             hsv_c = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#             H,S,V = hsv_c[:,:,0], hsv_c[:,:,1], hsv_c[:,:,2]
#             # กล้วยกว้างๆ + ตัดโทนพื้นหลังเด่น
#             dom_bg = _dominant_bg_hue_mask(hsv_c, keep_hue_ranges=((18,95),), bins=180, topk=2, tol=6)
#             banana_like = ((S>35)&(V>40)&(H>=18)&(H<=95)) & (~dom_bg)
#             banana_fg_ratio = float(np.mean(banana_like)) if banana_like.size else 0.0
#             if banana_fg_ratio < 0.20:
#                 # กล่องกินพื้นหลังเยอะ → ใช้ proposer
#                 need_proposer = True

#         if need_proposer:
#             print("[Proposer] YOLO unreliable/empty. Using color-based ROI...")
#             roi = _roi_from_hsv(img)
#             if roi is not None:
#                 x1r,y1r,x2r,y2r = roi
#                 roi_img = img[y1r:y2r, x1r:x2r]
#                 roi_results_list = _run_yolo(roi_img, conf=0.25, iou=0.50, imgsz=1280, augment=True, verbose=False)
#                 roi_results = roi_results_list[0]
#                 if len(roi_results.boxes) > 0:
#                     results = roi_results
#                     roi_offset = (x1r, y1r)
#                     print(f"[ROI] Using ROI boxes, offset={roi_offset}")

#         labels_info = []

#         if len(results.boxes) == 0:
#             print("No bananas found after two-pass + ROI. Saving original image.")
#             cv2.imwrite(output_path, img)
#             return [{"index": 0, "label": "No banana detected", "confidence": 0.0}], output_path

#         H_img, W_img = img.shape[:2]

#         for i, box in enumerate(results.boxes.data):
#             x1, y1, x2, y2 = map(int, box[:4])
#             conf = float(box[4])
#             cls_id = int(box[5])

#             if roi_offset != (0,0):
#                 x1 += roi_offset[0]; x2 += roi_offset[0]
#                 y1 += roi_offset[1]; y2 += roi_offset[1]

#             # clamp
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(W_img-1, x2), min(H_img-1, y2)

#             # เล็มขอบ 5% เพื่อตัดฉากหลัง
#             bw, bh = max(1, x2-x1), max(1, y2-y1)
#             padx, pady = max(1,int(0.05*bw)), max(1,int(0.05*bh))
#             xi1 = min(max(0, x1+padx), W_img-2)
#             yi1 = min(max(0, y1+pady), H_img-2)
#             xi2 = max(min(x2-padx, W_img-1), xi1+1)
#             yi2 = max(min(y2-pady, H_img-1), yi1+1)

#             crop = img[yi1:yi2, xi1:xi2]
#             if crop.size == 0:
#                 continue

#             hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#             H,S,V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

#             # --- ตัดพื้นหลังกวนออกก่อนเก็บสถิติ ---
#             dom_bg = _dominant_bg_hue_mask(hsv, keep_hue_ranges=((18,95),), bins=180, topk=2, tol=6)
#             base_mask = ((S > 35) & (V > 40) & (V < 250)) & (~dom_bg)

#             # ถ้ายังโล่ง ให้ผ่อนบ้าง
#             if not np.any(base_mask):
#                 base_mask = (S > 25) & (V > 35) & (~dom_bg)

#             m = base_mask > 0
#             if not np.any(m):
#                 m = np.ones_like(base_mask, dtype=bool)

#             H_vals, S_vals, V_vals = H[m], S[m], V[m]
#             H_med  = float(np.median(H_vals)) if H_vals.size else 0.0
#             S_mean = float(np.mean(S_vals)) if S_vals.size else 0.0
#             V_mean = float(np.mean(V_vals)) if V_vals.size else 0.0

#             dark_frac      = float(np.mean(V_vals < 80)) if V_vals.size else 0.0
#             very_dark_frac = float(np.mean(V_vals < 45)) if V_vals.size else 0.0
#             lowsat_frac    = float(np.mean(S_vals < 60)) if S_vals.size else 0.0

#             edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150)
#             edge_density = float(np.mean((edges > 0) & m)) if edges.size else 0.0

#             # ช่วงสี
#             H_YELLOW_MIN, H_YELLOW_MAX           = 20, 32
#             H_GREENYELLOW_MIN, H_GREENYELLOW_MAX = 32, 55
#             H_GREEN_MIN, H_GREEN_MAX             = 55, 85

#             green_mask  = ((H >= 38) & (H <= 85) & m)
#             yellow_mask = ((H >= 20) & (H <= 32) & m)
#             green_ratio  = float(np.mean(green_mask))
#             yellow_ratio = float(np.mean(yellow_mask))

#             # คำทำนายจากโมเดล
#             pred_label_raw = model.names[cls_id]
#             pred_label = pred_label_raw.strip().lower()
#             label = pred_label

#             print(f"[DBG #{i+1}] pred={pred_label} conf={conf:.2f}  "
#                   f"H_med={H_med:.1f} S={S_mean:.1f} V={V_mean:.1f}  "
#                   f"dark={dark_frac:.2f} vdark={very_dark_frac:.2f} edge={edge_density:.3f}  "
#                   f"green_ratio={green_ratio:.2f} yellow_ratio={yellow_ratio:.2f}")

#             # ===== ตรรกะเดิม (ตัดย่อ) + Guard เพิ่มเติม =====
#             strong_rotten = (
#                 (very_dark_frac >= 0.65 and H_med < 50)
#                 or ((dark_frac >= 0.75) and (lowsat_frac >= 0.60) and H_med < 50)
#                 or ((V_mean < 60) and (S_mean < 55) and (edge_density < 0.02) and H_med < 50)
#             )
#             strong_overripe = (dark_frac >= 0.35 and edge_density >= 0.02)

#             if strong_rotten:
#                 label = "rotten"
#             else:
#                 trust_model = (conf >= 0.70)

#                 # >>> NEW: ถ้าเขียวมากและเหลืองน้อย ให้ "บังคับ raw"
#                 if (green_ratio >= 0.60) and (yellow_ratio < 0.05) and (H_med >= H_GREEN_MIN):
#                     label = "raw"
#                 elif pred_label in ("raw", "unripe") and conf >= 0.55 and not strong_overripe:
#                     label = pred_label
#                 else:
#                     weak_rot_signals = (dark_frac < 0.25 and very_dark_frac < 0.08 and yellow_ratio < 0.07)
#                     if pred_label in ("rotten", "overripe") and weak_rot_signals and \
#                        ((green_ratio >= 0.55) or (H_GREENYELLOW_MIN <= H_med <= H_GREENYELLOW_MAX)):
#                         label = "unripe" if H_med < 58 else "raw"
#                     elif strong_overripe and conf < 0.60:
#                         label = "overripe"
#                     elif trust_model:
#                         label = pred_label
#                     # else คงเดิม

#             # refine raw vs unripe (ของเดิม)
#             if label in ("raw", "unripe"):
#                 strong_green = (green_ratio >= 0.75 and yellow_ratio < 0.04 and H_med >= H_GREEN_MIN)
#                 has_visible_yellow = (yellow_ratio >= 0.06) or (H_GREENYELLOW_MIN <= H_med <= H_GREENYELLOW_MAX)
#                 if strong_green:
#                     label = "raw"
#                 elif has_visible_yellow:
#                     label = "unripe"

#                 print(f"[REFINE #{i+1}] raw/unripe -> {label} "
#                       f"(H_med={H_med:.1f}, green={green_ratio:.2f}, yellow={yellow_ratio:.2f})")

#             # วาดผล
#             color = (255,255,255)
#             if label == "raw":       color = (0,255,255)
#             elif label == "unripe":  color = (0,165,255)
#             elif label == "ripe":    color = (0,255,0)
#             elif label == "overripe":color = (60,20,140)
#             elif label == "rotten":  color = (0,0,255)

#             cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
#             cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#             labels_info.append({"index": i+1, "label": label, "confidence": round(conf,2)})

#         cv2.imwrite(output_path, img)
#         print(f"Output image saved to {output_path}")
#         return labels_info, output_path

#     except Exception as e:
#         print(f"Error during banana detection: {e}")
#         return [], os.path.join("static", "error_processing.jpg")




import os
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------
# (ถ้าใช้บน Render/VM เล็กๆ แนะนำลด threads เพื่อลด RAM/latency)
# ---------------------------------------------------------
try:
    import torch
    torch.set_num_threads(1)  # ลดจำนวนเธรดของ PyTorch
except Exception:
    pass


# =========================
# Load model (robust version)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) ค่าเริ่มต้นให้ชัดเจนว่าอยู่ไฟล์เดียวกับสคริปต์
DEFAULT_MODEL = os.path.join(BASE_DIR, "best.pt")

# 2) อนุญาตให้ override ผ่าน ENV (เช่น MODEL_PATH=models/best.pt)
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL)

# ทำให้เป็น absolute เสมอ
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)

# candidate สำรอง (ช่วยกรณีวางไฟล์ไว้ที่อื่น)
CANDIDATES = [
    MODEL_PATH,
    DEFAULT_MODEL,
    os.path.join(BASE_DIR, "models", "best.pt"),
    os.path.join(BASE_DIR, "..", "best.pt"),
]

model = None
last_err = None
for p in CANDIDATES:
    try:
        if os.path.exists(p):
            model = YOLO(p)
            # บังคับ CPU (Render ฟรีไม่มี GPU)
            try:
                model.to("cpu")
            except Exception:
                pass
            print(f"[MODEL] Loaded YOLO weights from: {p}")
            break
    except Exception as e:
        last_err = e

if model is None:
    print(f"[MODEL] ERROR: Cannot load model from candidates: {CANDIDATES}")
    if last_err:
        print("Last error:", last_err)


# =========================
# YOLO wrapper
# =========================
def _run_yolo(img, conf=0.5, iou=0.45, imgsz=960, augment=False, verbose=False):
    """Wrapper เรียก predict ของ YOLO"""
    return model.predict(
        img, conf=conf, iou=iou, imgsz=imgsz, augment=augment, verbose=verbose
    )


def _roi_from_hsv(img):
    """
    หา ROI คร่าว ๆ ของกล้วยจากสีผิว (HSV) เพื่อลดพื้นหลัง
    คืนค่า: (x1,y1,x2,y2) หรือ None ถ้าหาไม่ได้
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    mask = ((S > 35) & (V > 35) & (V < 250)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # เผื่อขอบ 10%
    pad_x, pad_y = int(w * 0.1), int(h * 0.1)
    H_img, W_img = img.shape[:2]
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W_img - 1, x + w + pad_x)
    y2 = min(H_img - 1, y + h + pad_y)

    if (x2 - x1) < 64 or (y2 - y1) < 64:
        return None
    return (x1, y1, x2, y2)


# ---------- รวมกล่องแนวเส้นยาว (แก้ปัญหา 1 ลูก = 2 กล่อง) ----------
def merge_line_like_boxes(xyxy, conf, cls,
                          iou_thr=0.55, contain_thr=0.85,
                          ovlp_thr=0.65, gap_ratio=0.15):
    """
    รวมกล่องที่เป็นวัตถุชิ้นยาวชิ้นเดียวแต่ถูกแบ่งเป็น 2 กล่อง
    เกณฑ์รวม:
      - IoU สูง หรือ กล่องหนึ่งถูกอีกกล่องครอบ (containment)
      - หรือ overlap มากบนแกนหนึ่ง (x หรือ y) และช่องว่างตามแกนตั้งฉากเล็ก
    """
    if len(xyxy) <= 1:
        return xyxy, conf, cls

    xyxy = xyxy.astype(float).tolist()
    conf = conf.astype(float).tolist()
    cls  = cls.astype(int).tolist()

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
        area_b = max(0, bx2-bx1) * max(0, by2-by1)
        return inter / (area_a + area_b - inter + 1e-9)

    def containment(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        return inter / (area_a + 1e-9)

    def y_overlap(a, b):
        ay1, ay2 = a[1], a[3]; by1, by2 = b[1], b[3]
        inter = max(0, min(ay2, by2) - max(ay1, by1))
        return inter / (min(ay2 - ay1, by2 - by1) + 1e-9)

    def x_overlap(a, b):
        ax1, ax2 = a[0], a[2]; bx1, bx2 = b[0], b[2]
        inter = max(0, min(ax2, bx2) - max(ax1, bx1))
        return inter / (min(ax2 - ax1, bx2 - bx1) + 1e-9)

    def x_gap(a, b):
        ax1, ax2 = a[0], a[2]; bx1, bx2 = b[0], b[2]
        return max(0.0, max(ax1, bx1) - min(ax2, bx2))

    def y_gap(a, b):
        ay1, ay2 = a[1], a[3]; by1, by2 = b[1], b[3]
        return max(0.0, max(ay1, by1) - min(ay2, by2))

    changed = True
    while changed and len(xyxy) > 1:
        changed = False
        used = [False]*len(xyxy)
        new_xyxy, new_conf, new_cls = [], [], []

        order = np.argsort(-np.array(conf))  # คอนฟิเดนซ์สูงมาก่อน
        for i in order:
            if used[i]:
                continue
            ai = xyxy[i]
            best = ai[:]; best_conf = conf[i]; best_cls = cls[i]
            used[i] = True

            for j in order:
                if used[j] or i == j:
                    continue
                bj = xyxy[j]

                wi = max(1e-6, ai[2]-ai[0]); hi = max(1e-6, ai[3]-ai[1])
                wj = max(1e-6, bj[2]-bj[0]); hj = max(1e-6, bj[3]-bj[1])
                max_w, max_h = max(wi, wj), max(hi, hj)

                cond_iou = (iou(ai, bj) >= iou_thr) or \
                           (containment(ai, bj) >= contain_thr) or \
                           (containment(bj, ai) >= contain_thr)

                cond_horiz = (y_overlap(ai, bj) >= ovlp_thr) and \
                             (x_gap(ai, bj) <= gap_ratio * max_w)

                cond_vert  = (x_overlap(ai, bj) >= ovlp_thr) and \
                             (y_gap(ai, bj) <= gap_ratio * max_h)

                if cond_iou or cond_horiz or cond_vert:
                    best[0] = min(best[0], bj[0]); best[1] = min(best[1], bj[1])
                    best[2] = max(best[2], bj[2]); best[3] = max(best[3], bj[3])
                    best_conf = max(best_conf, conf[j])
                    used[j] = True
                    changed = True

            new_xyxy.append(best)
            new_conf.append(best_conf)
            new_cls.append(best_cls)

        xyxy, conf, cls = new_xyxy, new_conf, new_cls

    return np.array(xyxy, dtype=float), np.array(conf, dtype=float), np.array(cls, dtype=int)


# =========================
# Main detector
# =========================
def detect_banana(image_path):
    """
    ตรวจจับกล้วยและจำแนกระดับความสุก: raw, unripe, ripe, overripe, rotten
    return: (labels_info: list[dict], output_path: str)
    """
    if model is None:
        print("Error: YOLO model is not loaded. Cannot perform detection.")
        return [], os.path.join("static", "error_processing.jpg")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return [], os.path.join("static", "error_processing.jpg")

    os.makedirs("static", exist_ok=True)
    output_path = os.path.join("static", "result.jpg")

    try:
        # ---------- PASS 1 ----------
        results_list = _run_yolo(img, conf=0.5, iou=0.45, imgsz=960, augment=False, verbose=False)
        results = results_list[0]

        # ---------- PASS 2 ----------
        if len(results.boxes) == 0:
            print("[Two-Pass] Pass1 found 0 boxes. Retrying with lower conf & bigger imgsz...")
            results_list = _run_yolo(img, conf=0.25, iou=0.50, imgsz=1280, augment=True, verbose=False)
            results = results_list[0]

        # ---------- PASS 3 (ROI Fallback) ----------
        roi_offset = (0, 0)
        if len(results.boxes) == 0:
            print("[Two-Pass] Pass2 still 0 boxes. Trying ROI fallback...")
            roi = _roi_from_hsv(img)
            if roi is not None:
                x1r, y1r, x2r, y2r = roi
                roi_img = img[y1r:y2r, x1r:x2r]
                roi_results_list = _run_yolo(roi_img, conf=0.25, iou=0.50, imgsz=1280, augment=True, verbose=False)
                roi_results = roi_results_list[0]
                if len(roi_results.boxes) > 0:
                    results = roi_results
                    roi_offset = (x1r, y1r)
                    print(f"[ROI] Using ROI boxes, offset={roi_offset}")

        labels_info = []
        if len(results.boxes) == 0:
            print("No bananas found after two-pass + ROI. Saving original image.")
            cv2.imwrite(output_path, img)
            return [{"index": 0, "label": "No banana detected", "confidence": 0.0}], output_path

        H_img, W_img = img.shape[:2]

        # -------- ดึงกล่อง YOLO --------
        xyxy = results.boxes.xyxy.cpu().numpy().astype(float)
        conf = results.boxes.conf.cpu().numpy().astype(float)
        cls  = results.boxes.cls.cpu().numpy().astype(int)

        # ตัดกล่องเล็กจิ๋ว/เศษรอย
        wh = xyxy[:, 2:4] - xyxy[:, 0:2]
        min_side = max(12, 0.02 * max(H_img, W_img))  # 12px หรือ 2% ของภาพ
        keep_small = ~((wh[:, 0] < min_side) | (wh[:, 1] < min_side)) | (conf >= 0.60)
        xyxy, conf, cls = xyxy[keep_small], conf[keep_small], cls[keep_small]

        # รวมกล่องแนวเส้นยาว
        xyxy, conf, cls = merge_line_like_boxes(
            xyxy, conf, cls,
            iou_thr=0.55, contain_thr=0.85,
            ovlp_thr=0.65, gap_ratio=0.15
        )

        # ---------- วนทุกกล่องหลังรวม ----------
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            cconf = float(conf[i])
            cls_id = int(cls[i])

            if roi_offset != (0, 0):
                x1 += roi_offset[0]; x2 += roi_offset[0]
                y1 += roi_offset[1]; y2 += roi_offset[1]

            # clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W_img - 1, x2), min(H_img - 1, y2)

            # เล็มขอบ 5% เพื่อตัดฉากหลัง
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            padx, pady = max(1, int(0.05 * bw)), max(1, int(0.05 * bh))
            xi1 = min(max(0, x1 + padx), W_img - 2)
            yi1 = min(max(0, y1 + pady), H_img - 2)
            xi2 = max(min(x2 - padx, W_img - 1), xi1 + 1)
            yi2 = max(min(y2 - pady, H_img - 1), yi1 + 1)

            crop = img[yi1:yi2, xi1:xi2]
            if crop.size == 0:
                continue

            # ---------- วิเคราะห์สี/พื้นผิว ----------
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

            base_mask = ((S > 35) & (V > 40) & (V < 250)).astype(np.uint8)
            m = base_mask > 0
            if not np.any(m):
                m = np.ones_like(base_mask, dtype=bool)

            H_vals, S_vals, V_vals = H[m], S[m], V[m]
            H_med  = float(np.median(H_vals)) if H_vals.size else 0.0
            S_mean = float(np.mean(S_vals)) if S_vals.size else 0.0
            V_mean = float(np.mean(V_vals)) if V_vals.size else 0.0

            dark_frac      = float(np.mean(V_vals < 80)) if V_vals.size else 0.0
            very_dark_frac = float(np.mean(V_vals < 45)) if V_vals.size else 0.0
            lowsat_frac    = float(np.mean(S_vals < 60)) if S_vals.size else 0.0

            edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150)
            edge_density = float(np.mean((edges > 0) & m)) if edges.size else 0.0

            # ---- ช่วงสีโดยประมาณ (OpenCV H: 0–179)
            H_YELLOW_MIN, H_YELLOW_MAX           = 20, 32
            H_GREENYELLOW_MIN, H_GREENYELLOW_MAX = 32, 55
            H_GREEN_MIN, H_GREEN_MAX             = 55, 85

            # Green / Yellow coverage
            green_mask  = ((H >= 38) & (H <= 85) & m)     # ตัดเหลืองออกจากเขียว
            yellow_mask = ((H >= 20) & (H <= 32) & m)
            green_ratio  = float(np.mean(green_mask))
            yellow_ratio = float(np.mean(yellow_mask))

            # คำทำนายจากโมเดล
            pred_label_raw = model.names[cls_id]
            pred_label = pred_label_raw.strip().lower()
            label = pred_label

            print(f"[DBG #{i+1}] pred={pred_label} conf={cconf:.2f}  "
                  f"H_med={H_med:.1f} S={S_mean:.1f} V={V_mean:.1f}  "
                  f"dark={dark_frac:.2f} vdark={very_dark_frac:.2f} edge={edge_density:.3f}  "
                  f"green_ratio={green_ratio:.2f} yellow_ratio={yellow_ratio:.2f}")

            # ===== ตรรกะตัดสินใจหลัก =====
            strong_rotten = (
                (very_dark_frac >= 0.65 and H_med < 50)
                or ((dark_frac >= 0.75) and (lowsat_frac >= 0.60) and H_med < 50)
                or ((V_mean < 60) and (S_mean < 55) and (edge_density < 0.02) and H_med < 50)
            )
            strong_overripe = (dark_frac >= 0.35 and edge_density >= 0.02)

            if strong_rotten:
                label = "rotten"
            else:
                trust_model = (cconf >= 0.70)

                if pred_label in ("raw", "unripe") and cconf >= 0.55 and not strong_overripe:
                    label = pred_label
                else:
                    weak_rot_signals = (dark_frac < 0.25 and very_dark_frac < 0.08 and yellow_ratio < 0.07)

                    if pred_label in ("rotten", "overripe") and weak_rot_signals and \
                       (green_ratio >= 0.55 or (H_GREENYELLOW_MIN <= H_med <= H_GREENYELLOW_MAX)):
                        label = "unripe" if H_med < 58 else "raw"

                    elif strong_overripe and cconf < 0.60:
                        label = "overripe"

                    elif trust_model:
                        label = pred_label
                    # else: คง label เดิม

            # --- Final refinement: ชั่งน้ำหนัก raw vs unripe จากเขียว/เหลืองจริงบนผิว ---
            if label in ("raw", "unripe"):
                strong_green = (green_ratio >= 0.75 and yellow_ratio < 0.04 and H_med >= H_GREEN_MIN)
                has_visible_yellow = (yellow_ratio >= 0.06) or (H_GREENYELLOW_MIN <= H_med <= H_GREENYELLOW_MAX)

                if strong_green:
                    label = "raw"
                elif has_visible_yellow:
                    label = "unripe"

                print(f"[REFINE #{i+1}] raw/unripe -> {label} "
                      f"(H_med={H_med:.1f}, green={green_ratio:.2f}, yellow={yellow_ratio:.2f})")

            # ---------- วาดผล ----------
            color = (255, 255, 255)
            if label == "raw":
                color = (0, 255, 255)
            elif label == "unripe":
                color = (0, 165, 255)
            elif label == "ripe":
                color = (0, 255, 0)
            elif label == "overripe":
                color = (60, 20, 140)
            elif label == "rotten":
                color = (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} ({cconf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            labels_info.append({"index": i + 1, "label": label, "confidence": round(cconf, 2)})

        cv2.imwrite(output_path, img)
        print(f"Output image saved to {output_path}")
        return labels_info, output_path

    except Exception as e:
        print(f"Error during banana detection: {e}")
        return [], os.path.join("static", "error_processing.jpg")


# =========== CLI test ===========
if __name__ == "__main__":
    # ทดสอบรันจาก command line: python bana.py <path_to_image>
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bana.py <image_path>")
        sys.exit(0)

    labels, out_path = detect_banana(sys.argv[1])
    print("Results:", labels)
    print("Saved:", out_path)


