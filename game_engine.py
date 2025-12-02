"""
GameEngine - ゲームロジック管理クラス
時間経過に応じた画像処理とスコア計算，およびOpenCVによる自動クロップを担当
"""

import cv2
import os
import numpy as np
from image_processor import ImageProcessor


class GameEngine:
    """ゲームエンジンクラス"""

    def __init__(self, image_path, mode="blur", time_limit=30.0):
        """
        初期化
        """
        self.image_path = image_path
        self.mode = mode
        self.time_limit = time_limit

        self.original_image = None
        self.correct_answer = ""

        self.image_processor = ImageProcessor()

        # 1. 正解キーワードの抽出
        self.extract_answer_from_filename()

        # 2. 画像の読み込み
        self.load_image()

        # 3. OpenCVによる主要被写体の検出とクロップ
        # AIを使わず、画像処理で「被写体らしい場所」を特定します
        self.crop_to_main_subject()

    def load_image(self):
        """画像を読み込む"""
        if os.path.exists(self.image_path):
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                raise ValueError(f"画像の読み込みに失敗しました: {self.image_path}")
            # BGRからRGBに変換
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(f"画像ファイルが見つかりません: {self.image_path}")

    def extract_answer_from_filename(self):
        """ファイル名から正解キーワードを抽出"""
        filename = os.path.basename(self.image_path)
        name_without_ext = os.path.splitext(filename)[0]
        self.correct_answer = name_without_ext.split("_")[0].split("-")[0].lower()

    def crop_to_main_subject(self):
        """
        [修正版] 主要被写体を検出して画像をクロップする
        途切れたパーツも統合して、被写体全体を含むように調整
        """
        if self.original_image is None:
            return

        h, w = self.original_image.shape[:2]

        # 1. 前処理：グレースケール化とぼかし
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        # ノイズを消すため、ぼかしを少し強めに(9x9)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        # 2. エッジ検出 (Canny法)
        # 閾値を少し下げて(30, 150)、弱い輪郭（あごのラインなど）も拾いやすくする
        edges = cv2.Canny(blurred, 30, 150)

        # 3. 領域の結合 (膨張処理 - Dilation)
        # [重要] 回数を増やして(iterations=4)、離れたパーツ（目と口など）を強引にくっつける
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=4)

        # 4. 輪郭の抽出
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return

        # 5. 有効な輪郭をすべて集める
        valid_rects = []
        image_area = w * h

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch

            # 画像全体の0.5%以上の面積を持つものは「被写体の一部」とみなす
            # 小さすぎるゴミ（ノイズ）だけ除外
            if area > (image_area * 0.005):
                valid_rects.append((x, y, x + cw, y + ch))

        if not valid_rects:
            return

        # 6. すべての有効な輪郭を囲む「大きな外枠」を計算
        # min_x, min_y は最小値、max_x, max_y は最大値を取ることで全体を包含
        min_x = min([r[0] for r in valid_rects])
        min_y = min([r[1] for r in valid_rects])
        max_x = max([r[2] for r in valid_rects])
        max_y = max([r[3] for r in valid_rects])

        # 検出された幅と高さ
        detect_w = max_x - min_x
        detect_h = max_y - min_y

        # 画像の90%以上を占める場合は「背景ごと検出してしまった」とみなし、クロップしない
        if detect_w > w * 0.9 and detect_h > h * 0.9:
            return

        # 7. パディング（余白）を追加
        # 顔が見切れないよう、少し多めに余白(15%)を取る
        pad_x = int(detect_w * 0.15)
        pad_y = int(detect_h * 0.15)

        new_x1 = max(0, min_x - pad_x)
        new_y1 = max(0, min_y - pad_y)
        new_x2 = min(w, max_x + pad_x)
        new_y2 = min(h, max_y + pad_y)

        # クロップ実行
        cropped_img = self.original_image[new_y1:new_y2, new_x1:new_x2]

        # サイズチェック（極端に小さい画像にならないか確認）
        if (
            cropped_img.size > 0
            and cropped_img.shape[0] > 50
            and cropped_img.shape[1] > 50
        ):
            self.original_image = cropped_img

    def set_answer(self, answer):
        """正解を手動で設定"""
        self.correct_answer = answer.lower()

    def get_processed_image(self, elapsed_time):
        """
        経過時間に応じた現在の画像を取得
        """
        if self.original_image is None:
            return None

        # 進行度を計算
        if self.time_limit > 0:
            progress = elapsed_time / self.time_limit
        else:
            progress = 1.0
        progress = max(0.0, min(1.0, progress))

        # 画像処理の適用
        if self.mode == "blur":
            return self.image_processor.apply_blur(self.original_image, progress)
        elif self.mode == "zoom":
            return self.image_processor.apply_zoom(self.original_image, progress)
        elif self.mode == "hybrid":
            return self.image_processor.apply_hybrid(self.original_image, progress)
        else:
            return self.original_image.copy()

    def check_answer(self, user_answer):
        """回答をチェック"""
        user_answer_lower = user_answer.lower().strip()
        correct_answer_lower = self.correct_answer.lower().strip()

        is_correct = (
            user_answer_lower == correct_answer_lower
            or correct_answer_lower in user_answer_lower
            or user_answer_lower in correct_answer_lower
        )

        return is_correct, self.correct_answer

    def calculate_score(self, elapsed_seconds):
        """スコアを計算"""
        if elapsed_seconds >= self.time_limit:
            return 0
        if self.time_limit <= 0:
            return 0
        ratio_remaining = 1.0 - (elapsed_seconds / self.time_limit)
        score = 100 * ratio_remaining
        return max(0, score)

    def get_mode(self):
        """現在のモードを取得"""
        return self.mode

    def get_correct_answer(self):
        """正解を取得"""
        return self.correct_answer
