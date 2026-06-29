from ultralytics import YOLO

def main():
    # モデルのロード（v8なら yolov8n.pt, v11なら yolo11s.pt など）
    model = YOLO(r"models_load/yolo11s.pt")

    # 学習の実行
    results = model.train(
        data=r"C:/Users/kotan/gohara/cherry_yolo/annotated/raw_5goki_v2/data.yaml",
        epochs=200,        # 上限は多めに
        patience=40,       # 改善が止まれば自動停止（実質ここで決まる）
        imgsz=640,
        batch=45,        
        cache="disk",      # 読み込み高速化（メモリ次第でramのままでOK）
        cos_lr=True,       # 終盤の伸びを改善
        device=0,
        plots=True,
        project=r"C:/Users/kotan/gohara/cherry_yolo/models",  # 保存先フォルダ
        name="train_5goki_v2_11s",       # 実行名（同名が既存の場合は train2, train3 と自動連番）
    )

if __name__ == '__main__':
    main()