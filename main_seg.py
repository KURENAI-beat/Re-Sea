import glob
import os
import cv2
import folium
from folium.plugins import HeatMap
import numpy as np
from PIL import ExifTags, Image
import torch
from torchvision.ops import nms
from ultralytics import YOLO

# -------------------------------------------------------------
# 1. クラス定義・表示名マッピング
# -------------------------------------------------------------
UPCYCLABLE_CLASSES = {
    "plastic_bottle", "plastic_etc", "bottle", "cup", "bowl", "frisbee", "eel_trap"
}

FORCE_TRASH_KEYWORDS = [
    "buoy", "styrofoam", "net", "metal", "glass", "bag", "paper",
    "cardboard", "carboard", "cigarette", "wood", "driftwood", "clothes", "shoe"
]

def is_upcyclable(raw_name: str) -> bool:
    name = raw_name.lower().strip()
    if any(keyword in name for keyword in FORCE_TRASH_KEYWORDS):
        return False
    return name in UPCYCLABLE_CLASSES

CLASS_NAME_EN = {
    "plastic_bottle": "Bottle",
    "plastic_buoy": "Buoy",
    "plastic_etc": "Plastic_ETC",
    "styrofoam_box": "Styrofoam_Box",
    "styrofoam_piece": "Styrofoam_Piece",
    "net": "Net",
    "trash_net": "Net",
    "metal": "Metal",
    "metal_can": "Metal_Can",
    "glass": "Glass",
    "paper_cardboard": "Cardboard",
    "paper_carboard": "Cardboard",
    "plastic_bag": "Plastic_Bag",
    "bag": "Bag",
    "cigarette_butt": "Cigarette_Butt",
    "eel_trap": "Eel_Trap",
    "driftwood": "Driftwood",
    "wood": "Wood",
    "clothes": "Clothes",
    "shoe": "Shoe",
    "bottle": "Bottle",
    "cup": "Cup",
    "bowl": "Bowl",
    "frisbee": "Frisbee",
    "unknown": "Trash",
}

CLASS_NAME_JA = {
    "plastic_bottle": "ペットボトル",
    "plastic_buoy": "プラスチックブイ・浮子",
    "plastic_etc": "硬質プラ・キャップ・破片",
    "styrofoam_box": "発泡スチロール箱",
    "styrofoam_piece": "発泡スチロール破片",
    "net": "漁網・ロープ",
    "trash_net": "漁網・ロープ",
    "metal": "空き缶・金属",
    "metal_can": "空き缶",
    "glass": "ビン・ガラス",
    "paper_cardboard": "紙・ダンボール",
    "paper_carboard": "紙・ダンボール",
    "plastic_bag": "ビニール袋",
    "bag": "ビニール袋",
    "cigarette_butt": "吸い殻",
    "eel_trap": "うなぎの筒返し",
    "driftwood": "流木・木片",
    "wood": "流木・木材",
    "clothes": "衣類・布",
    "shoe": "靴・サンダル",
    "bottle": "ボトル",
    "cup": "カップ",
    "bowl": "ボウル",
    "frisbee": "フリスビー",
    "unknown": "その他ゴミ",
}


def get_exif_data(image_path):
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if not exif:
                return None

            gps_info = {}
            datetime_taken = "不明"

            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if tag_name == "DateTimeOriginal":
                    datetime_taken = value
                elif tag_name == "GPSInfo":
                    for gps_tag_id in value:
                        sub_tag = ExifTags.GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_info[sub_tag] = value[gps_tag_id]

            if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                lat_dms = gps_info["GPSLatitude"]
                lat = float(lat_dms[0]) + float(lat_dms[1]) / 60.0 + float(lat_dms[2]) / 3600.0
                if gps_info.get("GPSLatitudeRef") == "S":
                    lat = -lat

                lon_dms = gps_info["GPSLongitude"]
                lon = float(lon_dms[0]) + float(lon_dms[1]) / 60.0 + float(lon_dms[2]) / 3600.0
                if gps_info.get("GPSLongitudeRef") == "W":
                    lon = -lon

                if datetime_taken == "不明" and "GPSDateStamp" in gps_info:
                    date_str = str(gps_info["GPSDateStamp"]).replace(":", "/")
                    if "GPSTimeStamp" in gps_info:
                        t = gps_info["GPSTimeStamp"]
                        datetime_taken = f"{date_str} {int(t[0]):02d}:{int(t[1]):02d}:{int(t[2]):02d}"
                    else:
                        datetime_taken = date_str

                return {"lat": lat, "lon": lon, "datetime": datetime_taken}
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
    return None


# -------------------------------------------------------------
# 2. 高精度 重複除去＆包含関係フィルタ
# -------------------------------------------------------------
def filter_overlapping_and_contained_boxes(boxes, scores, classes, polygons, iou_thresh=0.25, contain_thresh=0.40):
    if len(boxes) == 0:
        return [], [], [], []

    boxes_t = torch.tensor(np.array(boxes), dtype=torch.float32)
    scores_t = torch.tensor(np.array(scores), dtype=torch.float32)

    keep_idx = nms(boxes_t, scores_t, iou_thresh).cpu().numpy()

    sorted_boxes = [boxes[i] for i in keep_idx]
    sorted_scores = [scores[i] for i in keep_idx]
    sorted_classes = [classes[i] for i in keep_idx]
    sorted_polygons = [polygons[i] for i in keep_idx]

    final_indices = []
    n = len(sorted_boxes)
    suppressed = [False] * n

    for i in range(n):
        if suppressed[i]:
            continue
        final_indices.append(i)
        box_a = sorted_boxes[i]
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])

        for j in range(i + 1, n):
            if suppressed[j]:
                continue
            box_b = sorted_boxes[j]
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

            inter_x1 = max(box_a[0], box_b[0])
            inter_y1 = max(box_a[1], box_b[1])
            inter_x2 = min(box_a[2], box_b[2])
            inter_y2 = min(box_a[3], box_b[3])

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            if inter_area > 0 and area_b > 0:
                overlap_on_b = inter_area / area_b
                overlap_on_a = inter_area / area_a if area_a > 0 else 0
                if overlap_on_b >= contain_thresh or overlap_on_a >= contain_thresh:
                    suppressed[j] = True

    return (
        [sorted_boxes[i] for i in final_indices],
        [sorted_scores[i] for i in final_indices],
        [sorted_classes[i] for i in final_indices],
        [sorted_polygons[i] for i in final_indices]
    )


# -------------------------------------------------------------
# 3. 多角形セグメンテーション対応 SAHIタイリング推論
# -------------------------------------------------------------
def predict_with_slicing_seg(model, img_cv2, slice_size=1280, overlap_ratio=0.2, conf=0.45, iou_thresh=0.25):
    h, w, _ = img_cv2.shape
    img_total_area = h * w

    all_boxes = []
    all_scores = []
    all_cls = []
    all_polygons = []

    def process_result(res, offset_x=0, offset_y=0, patch_w=1280, patch_h=1280):
        if res.boxes is None or len(res.boxes) == 0:
            return
        patch_area = patch_w * patch_h

        for idx, b in enumerate(res.boxes):
            score = float(b.conf[0])
            if score < conf:
                continue

            xyxy = b.xyxy[0].cpu().numpy()
            bw = xyxy[2] - xyxy[0]
            bh = xyxy[3] - xyxy[1]
            box_area = bw * bh

            if box_area > (patch_area * 0.30):
                continue
            if box_area > (img_total_area * 0.10):
                continue

            aspect_ratio = max(bw, bh) / max(1, min(bw, bh))
            if aspect_ratio > 10.0 and box_area < 2500:
                continue

            x1 = xyxy[0] + offset_x
            y1 = xyxy[1] + offset_y
            x2 = xyxy[2] + offset_x
            y2 = xyxy[3] + offset_y

            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(score)
            all_cls.append(int(b.cls[0]))

            if res.masks is not None and res.masks.xy is not None and idx < len(res.masks.xy):
                poly = res.masks.xy[idx].copy()
                if len(poly) >= 3:
                    poly[:, 0] += offset_x
                    poly[:, 1] += offset_y
                    all_polygons.append(poly)
                else:
                    all_polygons.append(None)
            else:
                all_polygons.append(None)

    stride = int(slice_size * (1 - overlap_ratio))
    x_starts = list(range(0, max(1, w - slice_size + stride), stride))
    y_starts = list(range(0, max(1, h - slice_size + stride), stride))
    if x_starts[-1] + slice_size < w:
        x_starts.append(w - slice_size)
    if y_starts[-1] + slice_size < h:
        y_starts.append(h - slice_size)

    for y in y_starts:
        for x in x_starts:
            pw = min(slice_size, w - x)
            ph = min(slice_size, h - y)
            patch = img_cv2[y:y + ph, x:x + pw]
            res = model(patch, imgsz=640, conf=conf, verbose=False)[0]
            process_result(res, offset_x=x, offset_y=y, patch_w=pw, patch_h=ph)

    res_full = model(img_cv2, imgsz=1280, conf=conf, iou=iou_thresh, verbose=False)[0]
    process_result(res_full, offset_x=0, offset_y=0, patch_w=w, patch_h=h)

    if not all_boxes:
        return None

    boxes_filtered, scores_filtered, classes_filtered, polygons_filtered = filter_overlapping_and_contained_boxes(
        all_boxes, all_scores, all_cls, all_polygons, iou_thresh=iou_thresh, contain_thresh=0.40
    )

    return {
        "boxes": boxes_filtered,
        "scores": scores_filtered,
        "classes": classes_filtered,
        "polygons": polygons_filtered,
        "names": model.names
    }


# -------------------------------------------------------------
# 4. 多角形（ポリゴンセグメンテーション）描画関数
# -------------------------------------------------------------
def draw_segmentation_results(img_cv2, det_result, save_path, alpha=0.45):
    if det_result is None or len(det_result["boxes"]) == 0:
        cv2.imwrite(save_path, img_cv2)
        return

    h, w, _ = img_cv2.shape
    img_out = img_cv2.copy()
    overlay = img_cv2.copy()

    boxes = det_result["boxes"]
    scores = det_result["scores"]
    classes = det_result["classes"]
    polygons = det_result["polygons"]
    names = det_result["names"]

    has_any_polygon = False

    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        cls_id = classes[i]
        raw_name = names.get(cls_id, str(cls_id)).lower()
        upcyclable = is_upcyclable(raw_name)

        color = (118, 230, 0) if upcyclable else (0, 0, 255)
        tag = "RECYCLE" if upcyclable else "TRASH"
        en_name = CLASS_NAME_EN.get(raw_name, raw_name.capitalize())
        label = f"[{tag}] {en_name} {score:.2f}"

        polygon_drawn = False
        if i < len(polygons) and polygons[i] is not None:
            polygon = polygons[i]
            if len(polygon) >= 3:
                poly_clipped = polygon.copy()
                poly_clipped[:, 0] = np.clip(poly_clipped[:, 0], 0, w - 1)
                poly_clipped[:, 1] = np.clip(poly_clipped[:, 1], 0, h - 1)
                pts = np.int32([poly_clipped])

                cv2.fillPoly(overlay, pts, color)
                cv2.polylines(img_out, pts, isClosed=True, color=color, thickness=2)
                polygon_drawn = True
                has_any_polygon = True

        x1, y1, x2, y2 = [int(v) for v in box]
        if not polygon_drawn:
            cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(th + 5, y1 - 6)
        cv2.rectangle(img_out, (x1, label_y - th - 5), (x1 + tw + 6, label_y + 3), color, -1)
        text_color = (0, 0, 0) if upcyclable else (255, 255, 255)
        cv2.putText(img_out, label, (x1 + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)

    if has_any_polygon:
        cv2.addWeighted(overlay, alpha, img_out, 1 - alpha, 0, img_out)

    cv2.imwrite(save_path, img_out)


# -------------------------------------------------------------
# 5. 地図（Folium）の生成関数
# -------------------------------------------------------------
def generate_folium_map(data_points, heat_data):
    """収集されたデータポイントからFoliumマップを生成して保存"""
    if not data_points:
        return

    avg_lat = sum(p["lat"] for p in data_points) / len(data_points)
    avg_lon = sum(p["lon"] for p in data_points) / len(data_points)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=14)

    HeatMap(heat_data, radius=25, blur=15).add_to(m)

    for p in data_points:
        breakdown_items = "".join([f"・{k}: <b>{v}</b> 個<br>" for k, v in p["breakdown"].items()])
        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 220px;">
            <h4 style="margin: 0 0 5px 0; color: #1e88e5;">📍 漂着ゴミ調査ポイント（多角形解析）</h4>
            <div style="font-size: 11px; color: #666; margin-bottom: 8px;">撮影日時: {p['datetime']}</div>
            <div style="background: #e8f5e9; padding: 6px; border-radius: 4px; margin-bottom: 6px;">
                <b style="color: #2e7d32;">♻️ 資源プラ: {p['upcyclable']} 個</b><br>
                <small style="color: #388e3c;">採取可能原料: 約 <b>{p['grams']}g</b><br>
                レジンアクセ: <b>約 {p['accessories']} 個分</b></small>
            </div>
            <div style="background: #ffebee; padding: 6px; border-radius: 4px; margin-bottom: 8px;">
                <b style="color: #c62828;">🗑️ 一般ゴミ: {p['other']} 個</b>
            </div>
            <details style="font-size: 12px; cursor: pointer;">
                <summary><b>📊 ゴミの詳細内訳（計 {p['total']} 個）</b></summary>
                <div style="margin-top: 4px; padding-left: 5px; max-height: 120px; overflow-y: auto;">
                    {breakdown_items}
                </div>
            </details>
        </div>
        """
        marker_color = "green" if p["upcyclable"] >= 10 else ("red" if p["total"] >= 15 else "blue")
        folium.Marker(
            location=[p["lat"], p["lon"]],
            popup=folium.Popup(popup_html, max_width=320),
            icon=folium.Icon(color=marker_color, icon="trash")
        ).add_to(m)

    # 2つのファイル名（多角形用・本番用）両方に自動保存！
    for target_path in ["beach_plastic_map_seg.html", "beach_plastic_map.html"]:
        m.save(target_path)
    print(f"🗺️ 地図を更新保存しました（現在ピン数: {len(data_points)} 箇所）: beach_plastic_map_seg.html / beach_plastic_map.html")


# -------------------------------------------------------------
# 6. メイン処理
# -------------------------------------------------------------
def main():
    candidate_models = [
        "v3_local_trash_seg.pt",
        "third_seg.pt",
        "best_seg.pt",
        "custom_seg.pt",
        "yolo26n-seg.pt",
        "yolov8n-seg.pt",
    ]
    model_path = None
    for candidate in candidate_models:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if not model_path:
        model_path = "yolov8n-seg.pt"

    print(f"使用モデル（多角形SAHI推論・高精度重複除去版）: {model_path}")
    model = YOLO(model_path)

    image_folder = "images"
    output_folder = "output_detected_seg"
    os.makedirs(output_folder, exist_ok=True)

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    image_files = sorted(list(set(image_files)))
    data_points = []
    heat_data = []

    print(f"{len(image_files)} 枚の画像を解析中（多角形SAHIタイリング推論・逐次地図更新）...")

    try:
        for idx, img_path in enumerate(image_files, 1):
            meta = get_exif_data(img_path)
            if not meta:
                print(f"GPS情報なし（スキップ）: {img_path}")
                continue

            img_cv2 = cv2.imread(img_path)
            if img_cv2 is None:
                continue

            det_result = predict_with_slicing_seg(model, img_cv2, slice_size=1280, overlap_ratio=0.2, conf=0.45, iou_thresh=0.25)

            save_path = os.path.join(output_folder, os.path.basename(img_path))
            draw_segmentation_results(img_cv2, det_result, save_path)

            upcyclable_count = 0
            other_count = 0
            class_counts = {}

            if det_result is not None:
                for cls_id in det_result["classes"]:
                    raw_name = det_result["names"].get(cls_id, str(cls_id)).lower()
                    ja_name = CLASS_NAME_JA.get(raw_name, raw_name)
                    class_counts[ja_name] = class_counts.get(ja_name, 0) + 1
                    if is_upcyclable(raw_name):
                        upcyclable_count += 1
                    else:
                        other_count += 1

            total_trash = upcyclable_count + other_count
            estimated_grams = upcyclable_count * 10
            estimated_accessories = max(1, round(estimated_grams / 5)) if upcyclable_count > 0 else 0

            data_points.append({
                "path": img_path,
                "lat": meta["lat"],
                "lon": meta["lon"],
                "datetime": meta["datetime"],
                "total": total_trash,
                "upcyclable": upcyclable_count,
                "other": other_count,
                "grams": estimated_grams,
                "accessories": estimated_accessories,
                "breakdown": class_counts
            })
            heat_data.append([meta["lat"], meta["lon"], total_trash])
            print(f"[{idx}/{len(image_files)}] 完了: {os.path.basename(img_path)} -> ゴミ総数: {total_trash}個 (資源プラ: {upcyclable_count}個 / アクセ約{estimated_accessories}個分)")

            # 5枚ごとに地図をリアルタイム自動保存（途中で中断してもそれまでのピンが確実に地図に残る！）
            if len(data_points) % 5 == 0:
                generate_folium_map(data_points, heat_data)

    except KeyboardInterrupt:
        print("\nユーザーによる中断を検知しました。そこまでの結果で地図を保存します...")

    finally:
        # 最終保存
        if data_points:
            generate_folium_map(data_points, heat_data)
            print("\n🎉 全ての解析結果を地図に保存しました！")


if __name__ == "__main__":
    main()
