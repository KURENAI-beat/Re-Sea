import os
import torch
from ultralytics import YOLO

def main():
    base_model = "yolov8n-seg.pt"
    print(f"ベースモデル: {base_model}（Instance Segmentation）")
    model = YOLO(base_model)

    data_yaml = "/Volumes/SSD/Re-Sea/resea-beach-seg/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"エラー: {data_yaml} が見つかりませんでした。")
        return

    print(f"データセット設定: {data_yaml}")

    # デバイス判定（Apple Silicon MPS / CPU）
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    try:
        model.train(
            data=data_yaml,
            epochs=30,
            imgsz=640,
            batch=8,
            name="resea_seg_v3",
            exist_ok=True,
            device=device
        )
    except Exception as e:
        print(f"デバイス {device} でのエラー: {e}")
        if device != "cpu":
            print("CPUモードで再試行します...")
            model.train(
                data=data_yaml,
                epochs=30,
                imgsz=640,
                batch=8,
                name="resea_seg_v3",
                exist_ok=True,
                device="cpu"
            )

    print("\n✅ 学習完了！")
    best_pt = "runs/segment/resea_seg_v3/weights/best.pt"
    print(f"成果物: {best_pt}")

if __name__ == "__main__":
    main()
