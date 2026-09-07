# Re:Sea

> 海岸漂着プラスチックの AI 検出 × GIS 可視化・アップサイクルプロジェクト

**Re:Sea（リシー）** は、地元の海岸（皆生・弓ヶ浜等）に漂着する海洋プラスチックを回収・データ化し、体験型アップサイクルワークショップや地域協創を展開するプロジェクトです。

スマートフォンで撮影した位置情報（GPS）付き写真から、物体検出 AI（YOLO）が漂着ゴミを自動検出し、インタラクティブな Web ヒートマップとして可視化します。

---

## 主な機能

* **AI 画像認識によるゴミ検出**  
  海岸写真からペットボトルやプラスチック破片、漁具などを自動認識・カウント。
* **Exif GPS 解析 & GIS マッピング**  
  撮影日時と緯度経度を自動抽出し、ゴミのホットスポットをヒートマップで可視化。
* **アップサイクル・ワークショップ連携**  
  データをもとに回収した海洋プラスチックを活用し、高専祭でのレジンアクセサリー制作体験や地域企業との連携を実施。

---

## システム構成

```text
[海岸での写真撮影 (広角・GPS ON)]
           │ (JPEG / PNG)
           ▼
[main.py (Python スクリプト)]
  ├─ Exif 解析 (撮影日時 & GPS 座標の抽出)
  └─ YOLO 推論 (海洋プラスチック・漂着ゴミの検出)
           │
           ▼
[beach_plastic_map.html (Folium)]
  └─ インタラクティブなヒートマップ & 詳細ピンの生成
           │
           ▼
[GitHub Pages / Web 公開]
```

---

## ディレクトリ構成

```text
beach_ai_project/
├── images/                # 解析対象の写真（GPS 付き画像）
├── output_detected/       # YOLO 検出結果画像（バウンディングボックス付き）
├── best.pt                # 学習済み海ゴミ検出モデル（配置時自動読み込み）
├── yolov8n.pt             # ベースモデル（フォールバック用）
├── main.py                # メイン実行スクリプト
├── beach_plastic_map.html # 出力される Web マップ
├── README.md              # 本ドキュメント
└── .gitignore
```

---

## 使用方法

### 1. 環境構築

Python 3.9 以上の環境で、仮想環境を作成してライブラリをインストールします。

```bash
# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate

# 必要パッケージのインストール
pip install ultralytics pillow folium
```

### 2. 写真の配置

スマートフォンで位置情報（GPS）をオンにして撮影した海岸写真を `images/` フォルダに配置します。

### 3. 解析の実行

```bash
python main.py
```

実行が完了すると、以下が出力されます：
* `output_detected/`: 検出結果のバウンディングボックスが描画された画像
* `beach_plastic_map.html`: ブラウザで開ける分布ヒートマップ

---

## 🛠️ 技術スタック

* **Language**: Python 3.9+
* **AI / Object Detection**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (YOLOv8 / YOLO26)
* **GIS / Mapping**: [Folium](https://python-visualization.github.io/folium/) / Leaflet.js
* **Image Processing**: Pillow (PIL)
* **Dataset / Training**: Roboflow Beach Trash Dataset / Google Colab (GPU)
