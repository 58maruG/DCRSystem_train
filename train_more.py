from ultralytics import YOLO

def main():
    # 前回の学習結果を初期重みとして読み込む
    model = YOLO(r"C:/Users/kotan/gohara/cherry_yolo/models/train_5goki_ver4_11n/weights/best.pt")

    results = model.train(
        data=r"C:/Users/kotan/gohara/cherry_yolo/annotated/raw_5goki_ver4/data.yaml",
        epochs=100,          # 追加で回したいエポック数
        patience=20,
        imgsz=640,
        device=0,
        plots=True,
        project=r"C:/Users/kotan/gohara/cherry_yolo/models",
        name="train_5goki_ver4_11n_more",  # 別名にして結果を分ける
    )

if __name__ == '__main__':
    main()