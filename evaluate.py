"""
学習済みYOLOモデルをtestデータで評価し、精度を測定するスクリプト。

- モデル: train_5goki_v4_11s/weights/best.pt
- テストデータ: raw_5goki_v4/data.yaml の test split (images/test)
"""

import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# ===== 設定 =====
MODEL_PATH = r"C:/Users/kotan/gohara/cherry_yolo/models/train_5goki_v4_11s/weights/best.pt"
DATA_YAML = r"C:/Users/kotan/gohara/cherry_yolo/annotated/raw_5goki_v4/data.yaml"
IMG_SIZE = 640
# GPU(CUDA)が使えれば 0、なければ CPU を自動選択
DEVICE = 0 if torch.cuda.is_available() else "cpu"
# ================


def main():
    # 入力ファイルの存在チェック（誤ったパスで実行した時に分かりやすくするため）
    if not Path(MODEL_PATH).is_file():
        print(f"[エラー] モデルが見つかりません: {MODEL_PATH}")
        sys.exit(1)
    if not Path(DATA_YAML).is_file():
        print(f"[エラー] data.yaml が見つかりません: {DATA_YAML}")
        sys.exit(1)

    try:
        # 1. 学習済みモデルをロード
        model = YOLO(MODEL_PATH)

        # 2. test split で評価を実行
        metrics = model.val(
            data=DATA_YAML,
            split="test",   # data.yaml の test: images/test を使用
            imgsz=IMG_SIZE,
            device=DEVICE,
            plots=True,     # PR曲線・混同行列などを画像で保存
        )
    except Exception as e:
        print(f"[エラー] 評価の実行中に問題が発生しました: {e}")
        sys.exit(1)

    # 3. 全体の精度を表示
    box = metrics.box
    print("\n================ 全体精度 ================")
    print(f"mAP50-95 : {box.map:.4f}")
    print(f"mAP50    : {box.map50:.4f}")
    print(f"mAP75    : {box.map75:.4f}")
    print(f"Precision: {box.mp:.4f}")
    print(f"Recall   : {box.mr:.4f}")

    # 4. クラスごとの精度を表示
    print("\n============== クラス別 mAP50-95 ==============")
    names = model.names  # {インデックス: クラス名}
    # box.ap_class_index には検出対象となったクラスのインデックスが入る
    for i, class_idx in enumerate(box.ap_class_index):
        class_name = names.get(int(class_idx), str(class_idx))
        print(f"{class_name:<14}: mAP50-95={box.maps[class_idx]:.4f}")

    # 結果の保存先（runs/detect/val* など）を案内
    print(f"\n結果の画像・グラフ保存先: {metrics.save_dir}")


if __name__ == "__main__":
    main()
