# YOLO モデル選定・改善 壁打ちメモ

このファイルは別端末の Claude Code でそのまま会話を継続するための引き継ぎ資料です。

---

## システム概要

- **システム名**: DCRSystem（サクランボ自動選果機）
- **フレームワーク**: Ultralytics YOLO のみ使用（他の YOLO 実装は使わない）
- **役割**: 流れてくるサクランボを4台のカメラでリアルタイム撮影し、不良を検出・排出する

### ハードウェア

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA RTX 4090 Laptop（VRAM 16GB）|
| CPU | Intel Core i9 |
| カメラ台数 | 4台（cam_top / cam_under / cam_inside / cam_outside）|
| カメラ解像度 | カメラごとに異なる（500×500 / 560×560 / 640×640）|
| 照明 | 固定・管理された環境（外光の影響なし）|
| 目標FPS | 20fps |

### 主なコードファイル

- アプリ本体: `C:\Users\kotan\gohara\DCRsystem_PC\DCRsystem5goki_PC_app_raw\`
- 学習スクリプト: `C:\Users\kotan\gohara\DCRsystem_PC\DCRSystem_train\`
- YOLOモジュール: `module_yolo_csv4_v3.py`

---

## 現在のモデル設定

```python
MODEL_PATH    = "Trained_Models/v4_11s.pt"   # YOLO11s（small）
YOLO_IMG_SIZE = 640                           # 全カメラ共通にリサイズ
CONF_THRESHOLD = 0.5
PREDICT_CONF   = 0.1   # ByteTracker への入力閾値（最終判定は CONF_THRESHOLD）
```

### 推論アーキテクチャ

- **バックグラウンドスレッド**で `model.predict` を非同期実行（GUI ブロック防止）
- **ByteTracker** でカメラごとに個体追跡
- HSV マスクで「サクランボが中央帯を通過中」の間だけ推論を投入（無駄な推論を排除）
- 4カメラ共通の推論ワーカー1スレッド

---

## 検出クラス一覧

```python
COLORS = {
    "birddamage", "healthy", "mold", "stemcrack", "twin",
    "unripe", "malformation", "crack", "wilt", "suturecrack",
    "brownrot", "blacktwin", "kasure", "insect"
}
```

---

## 実測パフォーマンス（ログより：2026-06-23）

ログフォルダ: `logs_cycle_5goki/cycle_20260623.csv`

| 指標 | 実測値 |
|---|---|
| 推論時間（infer_latency） | **11〜13ms**（平均約12ms）|
| 前処理（preproc） | 0.5ms |
| 後処理（postproc） | 0.28ms |
| カメラ取得レイテンシ | **48〜49ms**（≒ 20fps）|
| GPU使用率 | **8〜15%**（ほぼ遊んでいる）|
| GPU VRAM（モデル） | **約71MB**（16GBのうち0.4%）|
| GPU電力 | 35〜43W（最大の半分以下）|

**重要な気づき**: ボトルネックはカメラ取得（49ms）であって推論ではない。
GPU はほぼ遊んでいるため、モデルを大きくしても20fps 維持は問題ない。

---

## クラス別精度（YOLO11n での検証結果、11s とほぼ同等）

ファイル: `v4_11n.txt`（200 epochs 学習後の val 結果）

| クラス | mAP50 | mAP50-95 | P | R | 評価 |
|---|---|---|---|---|---|
| **insect** | 0.052 | 0.017 | 0.171 | 0.070 | **壊滅的** |
| **kasure** | 0.672 | 0.372 | 0.839 | 0.608 | 深刻 |
| **wilt** | 0.735 | 0.475 | 0.944 | 0.629 | 悪い |
| SutureCrack | 0.836 | 0.511 | 0.788 | 0.795 | やや悪い |
| crack | 0.822 | 0.581 | 0.724 | 0.789 | やや悪い |
| BirdDamage | 0.845 | 0.522 | 0.977 | 0.653 | やや悪い |
| BrownRot | 0.903 | 0.695 | 0.919 | 0.880 | 良好 |
| blacktwin | 0.960 | 0.751 | 0.907 | 0.936 | 良好 |
| malformation | 0.951 | 0.829 | 0.875 | 0.935 | 良好 |
| mold | 0.927 | 0.626 | 0.861 | 0.870 | 良好 |
| StemCrack | 0.884 | 0.663 | 0.742 | 0.841 | 良好 |
| twin | 0.994 | 0.994 | 0.991 | 0.996 | 優秀 |
| unripe | 0.986 | 0.986 | 0.927 | 0.957 | 優秀 |
| healthy | 0.979 | 0.979 | 0.896 | 0.972 | 優秀 |

---

## 問題の原因分析

### insect（壊滅的）
- 学習データ: **59枚・71インスタンス**（圧倒的に少ない）
- P=0.17, R=0.07 → ほぼ学習できていない
- **今シーズンはこれ以上データ収集不可**（収穫シーズン終了）
- モデルを大きくするだけでは解決しない → データレベルの対策が必要

### kasure / wilt（Recall 低下 = 見落とし多い）
- wilt: P=0.944（検出時は正確）なのに R=0.629（半分以上見落とし）
- BirdDamage も同様（P高・R低）
- 原因: データの多様性不足、または他クラスとの視覚的混同

### crack / SutureCrack（P・R ともに中程度）
- 視覚的に微細な特徴 → **大きいモデルへの変更が効きやすい**

---

## モデル選定の結論

### 試す優先順位

| 優先度 | モデル | 理由 |
|---|---|---|
| 1 | **YOLO11m** | GPU に余裕あり、crack/SutureCrack/kasure の改善期待 |
| 2 | **YOLO11l** | 11m で改善が見られたら次に試す |
| 3 | YOLO11x | 試す価値あり（GPU が余裕すぎるため）|

### 試さなくてよいもの

| モデル | 理由 |
|---|---|
| YOLO11n | 現在の11s より小さい。精度ダウン確定 |
| YOLO11-cls | 画像全体を1クラスに分類するタスク。このシステムに合わない |
| YOLO-pose / OBB | サクランボには不要 |
| YOLO11-seg | twin/blacktwin 改善には有効だが、アノテーションをポリゴンマスクに作り直す必要があり工数大 |

---

## 今すぐできる改善策

### insect 対策（データなしで精度を上げる）

**① オーバーサンプリング（最も手軽）**

insect の画像・ラベルファイルをコピーして3〜5倍に増やしてからデータセットに追加する。
YOLO はファイル数ベースでサンプリングするため、学習頻度が上がる。

```
datasets/images/train/
  insect_001.jpg
  insect_001_copy2.jpg   ← 同じ画像をコピー
  insect_001_copy3.jpg
```

**② copy_paste augmentation（Ultralytics 標準機能）**

insect のバウンディングボックス領域を他画像に貼り付けて合成データを自動生成する。

```python
model.train(
    ...
    copy_paste=0.5,  # insect に特に効果的
    mixup=0.2,
    degrees=15,
    fliplr=0.5,
    flipud=0.3,
)
```

**③ エポック数を増やす**

200 → 400〜500 epochs に延長。RTX4090 なら時間的に問題なし。

### 全体的な改善策

| 優先度 | やること | 対象クラス |
|---|---|---|
| 1 | insect 画像を3〜5倍コピーしてデータセット追加 | insect |
| 2 | `copy_paste=0.5`, `mixup=0.2` を学習引数に追加 | 全体（insect 特効）|
| 3 | YOLO11m に変更して再学習（400〜500 epochs） | kasure / wilt / crack |

---

## 次のアクション（未着手）

- [ ] 現在の `train.py` を確認し、augmentation 引数を追加
- [ ] insect 画像のオーバーサンプリング実施
- [ ] YOLO11m で再学習・val 結果を 11s と比較
- [ ] 改善が見られた場合、11l も試す

---

## 補足：STRICT_THRESHOLDS（本番システムの判定閾値）

```python
STRICT_THRESHOLDS = {
    "stemcrack":    0.8,
    "crack":        0.8,
    "birddamage":   0.8,
    "twin":         0.9,
    "blacktwin":    0.9,
    "malformation": 0.9,
    "unripe":       0.9,
}
# insect はここに含まれていない → conf 0.5 以上で常に採用
```
