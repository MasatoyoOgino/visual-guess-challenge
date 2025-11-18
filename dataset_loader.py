"""
データセットローダー
imagesフォルダからランダムに画像を選択する機能を提供
"""

import os
import random
from pathlib import Path


class DatasetLoader:
    """データセットローダークラス"""

    def __init__(self, images_dir="images"):
        """
        初期化

        Args:
            images_dir: 画像フォルダのパス
        """
        self.images_dir = images_dir
        self.supported_formats = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        self.image_files = []
        self.load_image_list()

    def load_image_list(self):
        """画像ファイルのリストを読み込む"""
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir, exist_ok=True)
            return

        self.image_files = [
            os.path.join(self.images_dir, f)
            for f in os.listdir(self.images_dir)
            if os.path.isfile(os.path.join(self.images_dir, f))
            and Path(f).suffix.lower() in self.supported_formats
        ]

    def get_random_image(self):
        """
        ランダムに画像を1枚選択

        Returns:
            画像ファイルのパス。画像がない場合はNone
        """
        if not self.image_files:
            return None

        return random.choice(self.image_files)

    def get_all_images(self):
        """
        すべての画像ファイルのリストを取得

        Returns:
            画像ファイルパスのリスト
        """
        return self.image_files.copy()

    def get_image_count(self):
        """画像の総数を取得"""
        return len(self.image_files)

    def get_images_by_category(self):
        """
        カテゴリ別に画像を分類

        Returns:
            {カテゴリ名: [画像パスのリスト]} の辞書
        """
        categories = {}

        for image_path in self.image_files:
            filename = os.path.basename(image_path)
            # ファイル名の最初の部分をカテゴリ名とする
            category = filename.split("_")[0].split("-")[0].lower()

            if category not in categories:
                categories[category] = []
            categories[category].append(image_path)

        return categories

    def refresh(self):
        """画像リストを再読み込み"""
        self.load_image_list()
