"""
GameEngine - ゲームロジック管理クラス
時間経過に応じた画像処理（線形変換）とスコア計算を担当
"""

import cv2
import os
from image_processor import ImageProcessor


class GameEngine:
    """ゲームエンジンクラス"""

    def __init__(self, image_path, mode="blur", time_limit=30.0):
        """
        初期化

        Args:
            image_path: 画像ファイルのパス
            mode: ゲームモード ('blur', 'zoom', 'hybrid')
            time_limit: 画像が完全にクリアになるまでの時間（秒）
        """
        self.image_path = image_path
        self.mode = mode
        self.time_limit = time_limit
        self.original_image = None
        self.correct_answer = ""

        # 画像プロセッサのインスタンス
        self.image_processor = ImageProcessor()

        # 画像の読み込み
        self.load_image()

        # 正解キーワードの抽出（ファイル名から）
        self.extract_answer_from_filename()

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
        # 拡張子を除く
        name_without_ext = os.path.splitext(filename)[0]
        # アンダースコアやハイフンで分割して最初の部分を正解とする
        self.correct_answer = name_without_ext.split("_")[0].split("-")[0].lower()

    def set_answer(self, answer):
        """正解を手動で設定"""
        self.correct_answer = answer.lower()

    def get_processed_image(self, elapsed_time):
        """
        経過時間に応じた現在の画像を取得

        Args:
            elapsed_time: 経過時間（秒）

        Returns:
            処理された画像
        """
        if self.original_image is None:
            return None

        # 進行度を計算 (0.0:開始直後 -> 1.0:完了)
        if self.time_limit > 0:
            progress = elapsed_time / self.time_limit
        else:
            progress = 1.0

        # 0.0〜1.0の範囲にクリップ
        progress = max(0.0, min(1.0, progress))

        # ImageProcessorには progress (0.0-1.0) を渡す
        if self.mode == "blur":
            return self.image_processor.apply_blur(self.original_image, progress)
        elif self.mode == "zoom":
            return self.image_processor.apply_zoom(self.original_image, progress)
        elif self.mode == "hybrid":
            return self.image_processor.apply_hybrid(self.original_image, progress)
        else:
            return self.original_image.copy()

    def check_answer(self, user_answer):
        """
        回答をチェック

        Args:
            user_answer: ユーザーの回答

        Returns:
            (is_correct, correct_answer) のタプル
        """
        user_answer_lower = user_answer.lower().strip()
        correct_answer_lower = self.correct_answer.lower().strip()

        # 完全一致または部分一致で判定
        is_correct = (
            user_answer_lower == correct_answer_lower
            or correct_answer_lower in user_answer_lower
            or user_answer_lower in correct_answer_lower
        )

        return is_correct, self.correct_answer

    def calculate_score(self, elapsed_seconds):
        """
        スコアを計算

        Args:
            elapsed_seconds: 経過時間（秒）

        Returns:
            スコア（0-100点）
        """
        # 制限時間を超えていたら0点
        if elapsed_seconds >= self.time_limit:
            return 0

        if self.time_limit <= 0:
            return 0

        # 残り時間の割合をスコアとする (100点満点)
        # 時間経過とともに 100 -> 0 に線形で減少
        ratio_remaining = 1.0 - (elapsed_seconds / self.time_limit)
        score = 100 * ratio_remaining

        return max(0, score)

    def get_mode(self):
        """現在のモードを取得"""
        return self.mode

    def get_correct_answer(self):
        """正解を取得"""
        return self.correct_answer
