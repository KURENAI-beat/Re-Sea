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
# レジンアクセサリーの原料に活用できる資源プラスチック（うなぎの筒返しも硬質プラ原料として活用）
UPCYCLABLE_CLASSES = {
    "plastic_bottle", "plastic_etc", "bottle", "cup", "bowl", "frisbee", "eel_trap"
}

# 資源プラ混入を防ぐ強制ゴミキーワード（ブイ・発泡スチロール・漁網・金属・袋・木材・衣類・靴等）
FORCE_TRASH_KEYWORDS = [
    "buoy", "styrofoam", "net", "metal", "glass", "bag", "paper",
    "cardboard", "carboard", "cigarette", "wood", "driftwood", "clothes", "shoe"
]

def is_upcyclable(raw_name: str) -> bool:
    """資源プラスチック（レジン原料）として適格か判定（ブイや発泡スチロール・木材等は厳格に除外）"""
    name = raw_name.lower().strip()
    if any(keyword in name for keyword in FORCE_TRASH_KEYWORDS):
        return False
    return name in UPCYCLABLE_CLASSES

# OpenCV描画用（英数字統一で文字化け防止）
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

# Webマップ（HTML）用日本語表示マッピング
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
# 2. SAHI（スライシング・タイリング推論）
# -------------------------------------------------------------
def predict_with_slicing(model, img_cv2, slice_size=1280, overlap_ratio=0.2, conf=0.40, iou_thresh=0.45):
    h, w, _ = img_cv2.shape
    if w <= slice_size and h <= slice_size:
        results = model(img_cv2, imgsz=slice_size, conf=conf, iou=iou_thresh, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        return {
            "boxes": boxes.xyxy.cpu().numpy(),
            "scores": boxes.conf.cpu().numpy(),
            "classes": boxes.cls.cpu().numpy().astype(int),
            "names": model.names
        }

    stride = int(slice_size * (1 - overlap_ratio))
    x_starts = list(range(0, max(1, w - slice_size + stride), stride))
    y_starts = list(range(0, max(1, h - slice_size + stride), stride))
    if x_starts[-1] + slice_size < w:
        x_starts.append(w - slice_size)
    if y_starts[-1] + slice_size < h:
        y_starts.append(h - slice_size)

    all_boxes, all_scores, all_cls = [], [], []

    # 各パッチ（タイル）の推論
    for y in y_starts:
        for x in x_starts:
            patch = img_cv2[y:y + slice_size, x:x + slice_size]
            res = model(patch, imgsz=640, conf=conf, verbose=False)
            boxes = res[0].boxes
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    all_boxes.append([xyxy[0] + x, xyxy[1] + y, xyxy[2] + x, xyxy[3] + y])
                    all_scores.append(float(b.conf[0]))
                    all_cls.append(int(b.cls[0]))

    # 全体画像推論（大きなゴミ用）も合算
    res_full = model(img_cv2, imgsz=1280, conf=conf, iou=iou_thresh, verbose=False)
    if res_full[0].boxes is not None and len(res_full[0].boxes) > 0:
        for b in res_full[0].boxes:
            all_boxes.append(b.xyxy[0].cpu().numpy())
            all_scores.append(float(b.conf[0]))
            all_cls.append(int(b.cls[0]))

    if not all_boxes:
        return None

    boxes_t = torch.tensor(np.array(all_boxes), dtype=torch.float32)
    scores_t = torch.tensor(np.array(all_scores), dtype=torch.float32)
    cls_t = torch.tensor(np.array(all_cls), dtype=torch.int64)

    keep_idx = nms(boxes_t, scores_t, iou_thresh)
    return {
        "boxes": boxes_t[keep_idx].numpy(),
        "scores": scores_t[keep_idx].numpy(),
        "classes": cls_t[keep_idx].numpy(),
        "names": model.names
    }


# -------------------------------------------------------------
# 3. 資源プラ(緑枠: RECYCLE) vs 一般ゴミ(赤枠: TRASH) の色分け描画
# -------------------------------------------------------------
def draw_custom_boxes(img_cv2, det_result, save_path):
    if det_result is None or len(det_result["boxes"]) == 0:
        cv2.imwrite(save_path, img_cv2)
        return

    img_out = img_cv2.copy()
    names = det_result["names"]

    for box, score, cls_id in zip(det_result["boxes"], det_result["scores"], det_result["classes"]):
        x1, y1, x2, y2 = [int(v) for v in box]
        raw_name = names.get(cls_id, str(cls_id)).lower()
        upcyclable = is_upcyclable(raw_name)

        # 資源プラは鮮やかな緑 (BGR: 0, 230, 118)、一般ゴミは赤 (BGR: 0, 0, 255)
        color = (118, 230, 0) if upcyclable else (0, 0, 255)
        tag = "RECYCLE" if upcyclable else "TRASH"
        en_name = CLASS_NAME_EN.get(raw_name, raw_name.capitalize())
        label = f"[{tag}] {en_name} {score:.2f}"

        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_out, (x1, max(0, y1 - 25)), (x1 + tw + 6, max(0, y1)), color, -1)
        cv2.putText(img_out, label, (x1 + 3, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0) if upcyclable else (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(save_path, img_out)


# -------------------------------------------------------------
# 4. 地図（Folium）の生成関数
# -------------------------------------------------------------
def generate_folium_map(data_points, heat_data):
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
            <h4 style="margin: 0 0 5px 0; color: #1e88e5;">📍 漂着ゴミ調査ポイント</h4>
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

    m.save("beach_plastic_map.html")
    print(f"🗺️ 地図を更新保存しました（現在ピン数: {len(data_points)} 箇所）: beach_plastic_map.html")


# -------------------------------------------------------------
# 5. メイン処理
# -------------------------------------------------------------
def main():
    # 利用可能なモデルを優先度順に探索
    for candidate in [
        "v2_sea_trash_buoy_box.pt",
        "second.pt",
        "v1_general_trash_box.pt",
        "first.pt",
        "yolo26n.pt",
        "yolov8n.pt",
    ]:
        if os.path.exists(candidate):
            model_path = candidate
            break
    print(f"使用モデル: {model_path}")
    model = YOLO(model_path)

    image_folder = "images"
    output_folder = "output_detected"
    os.makedirs(output_folder, exist_ok=True)

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    image_files = sorted(list(set(image_files)))
    data_points = []
    heat_data = []

    print(f"{len(image_files)} 枚の画像を解析中（四角形SAHIタイリング推論・本番運用）...")

    try:
        for idx, img_path in enumerate(image_files, 1):
            meta = get_exif_data(img_path)
            if not meta:
                print(f"GPS情報なし（スキップ）: {img_path}")
                continue

            img_cv2 = cv2.imread(img_path)
            if img_cv2 is None:
                continue

            det_result = predict_with_slicing(model, img_cv2, slice_size=1280, overlap_ratio=0.2, conf=0.40, iou_thresh=0.45)
            
            save_path = os.path.join(output_folder, os.path.basename(img_path))
            draw_custom_boxes(img_cv2, det_result, save_path)

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

            # 5枚ごとに地図を自動保存
            if len(data_points) % 5 == 0:
                generate_folium_map(data_points, heat_data)

    except KeyboardInterrupt:
        print("\nユーザーによる中断を検知しました。そこまでの結果で地図を保存します...")

    finally:
        if data_points:
            generate_folium_map(data_points, heat_data)
            print("\n🎉 全ての解析結果を本番地図に保存完了: beach_plastic_map.html")


if __name__ == "__main__":
    main()
