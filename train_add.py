from ultralytics import YOLO

def main():
    # 前回学習済みのベストモデルから追加学習を開始
    model = YOLO(r"C:\Users\kotan\gohara\cherry_yolo\models\train_5goki_ver4_11n2\weights\best.pt")

    results = model.train(
        # ★重要: 追加データを含む「全クラス」のデータを必ず使う
        #  新データのみで学習すると既存クラスの精度が劣化する（破滅的忘却）
        #  新しい画像を追加した場合は、このdata.yamlに反映しておくこと
        data=r"C:/Users/kotan/gohara/cherry_yolo/annotated/raw_5goki_ver4/data.yaml",
        epochs=50,
        imgsz=640,
        device=0,
        plots=True,

        # --- 既存クラスの精度劣化を防ぐ設定 ---
        freeze=10,          # バックボーン前10層を凍結（healthy/unripe/twins等の特徴抽出能力を保持）
        lr0=0.001,          # 学習率を低く抑える（元の重みを急激に上書きしない）
        lrf=0.01,           # 最終学習率 = lr0 × lrf = 0.00001
        cos_lr=True,        # コサイン学習率スケジューリングで緩やかに収束させる
        warmup_epochs=2,    # ウォームアップは短めに（fine-tuningなので不要）
        patience=50,        # 50エポック改善なければ早期停止（過学習防止）

        # --- 少数クラス（crack, birddamage）の補強 ---
        copy_paste=0.3,     # 少数クラスの物体を他画像に貼り付けて自動水増し

        batch=8,            # imgsz=640では小さいバッチが必要（OOMが出る場合は4に下げる）
        project=r"C:/Users/kotan/gohara/cherry_yolo/models",  # 保存先フォルダ
        name="train_5goki_ver4_add",   # 実行名（同名が既存の場合は train_add2, train_add3 と自動連番）
    )

if __name__ == '__main__':
    main()