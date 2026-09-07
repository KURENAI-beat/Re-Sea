import glob
import os
import folium
from folium.plugins import HeatMap
from PIL import Image, ExifTags
from ultralytics import YOLO


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

                # iPhone写真等で通常Exifに日時がない場合、GPSタイムスタンプから補完
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


def main():
    # 学習済み専用モデルがあれば使用、なければベースモデルを使用
    model_path = "first.pt" if os.path.exists("first.pt") else "yolov8n.pt"
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

    print(f"{len(image_files)} 枚の画像を解析中...")

    for img_path in image_files:
        meta = get_exif_data(img_path)
        if not meta:
            print(f"GPS情報なし（スキップ）: {img_path}")
            continue

        # 物体検出（高解像度推論1280px、閾値0.40で微小・変形ゴミを捕捉）
        results = model(
            img_path,
            imgsz=1280,
            conf=0.40,
            iou=0.65,
            max_det=1000,
            verbose=False,
        )        
        boxes = results[0].boxes
        
        # 検出ゴミ総数のカウント（専用モデルの場合は全検出ボックス、初期モデルの場合は関連クラス集計）
        if boxes is not None:
            if model_path == "first.pt":
                trash_count = len(boxes)
            else:
                # 初期モデル(COCO)用: 39:bottle, 41:cup, 45:bowl, 29:frisbee
                valid_classes = [39, 41, 45, 29]
                trash_count = len([int(c) for c in boxes.cls if int(c) in valid_classes])
        else:
            trash_count = 0

        save_path = os.path.join(output_folder, os.path.basename(img_path))
        results[0].save(save_path)

        data_points.append({
            "path": img_path,
            "lat": meta["lat"],
            "lon": meta["lon"],
            "datetime": meta["datetime"],
            "count": trash_count
        })
        heat_data.append([meta["lat"], meta["lon"], trash_count])
        print(f"完了: {os.path.basename(img_path)} -> 検出ゴミ数: {trash_count}個")

    if not data_points:
        print("位置情報付きの画像データがありませんでした。")
        return

    avg_lat = sum(p["lat"] for p in data_points) / len(data_points)
    avg_lon = sum(p["lon"] for p in data_points) / len(data_points)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=14)

    HeatMap(heat_data, radius=25, blur=15).add_to(m)

    for p in data_points:
        popup_text = f"""
        <b>撮影日時:</b> {p['datetime']}<br>
        <b>検出ゴミ数:</b> {p['count']} 個<br>
        <b>座標:</b> {p['lat']:.5f}, {p['lon']:.5f}
        """
        folium.Marker(
            location=[p["lat"], p["lon"]],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color="red" if p["count"] > 5 else "blue", icon="trash")
        ).add_to(m)

    m.save("beach_plastic_map.html")
    print("マップ出力完了: beach_plastic_map.html")


if __name__ == "__main__":
    main()
