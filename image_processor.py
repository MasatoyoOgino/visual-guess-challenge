"""
ImageProcessor - 画像処理クラス
ぼかし処理、ズーム処理、ハイブリッド処理を担当
"""

import cv2
import numpy as np


class ImageProcessor:
    """画像プロセッサークラス"""

    def __init__(self):
        """初期化"""
        pass

    def apply_blur(self, image, progress):
        """
        progress: 0.0 (開始) -> 1.0 (クリア)
        """
        if image is None:
            return None

        # 進行度を 0.0-1.0 にクリップ
        progress = max(0.0, min(1.0, progress))

        # 最大ぼかし強度 (sigma)
        max_sigma = 30.0

        # 進行度に応じてsigmaを減少 (1.0のとき0になる)
        sigma = max_sigma * (1.0 - progress)

        if sigma <= 0.1:  # ほぼ0なら処理しない
            return image.copy()

        # カーネルサイズをsigmaから計算 (奇数にする必要がある)
        ksize = int(sigma * 6) + 1
        if ksize % 2 == 0:
            ksize += 1

        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    def apply_zoom(self, image, progress):
        """
        progress: 0.0 (開始) -> 1.0 (クリア)
        """
        if image is None:
            return None

        height, width = image.shape[:2]
        progress = max(0.0, min(1.0, progress))

        # 最小表示割合 (例: 12.5% = 1/8)
        min_ratio = 0.125

        # 線形補間: min_ratio から 1.0 へ変化
        current_ratio = min_ratio + (1.0 - min_ratio) * progress

        # 切り出しサイズ計算
        crop_width = int(width * current_ratio)
        crop_height = int(height * current_ratio)

        # 中心座標
        cx, cy = width // 2, height // 2

        # 切り出し範囲 (範囲外に出ないようクリップ)
        x1 = max(0, cx - crop_width // 2)
        y1 = max(0, cy - crop_height // 2)
        x2 = min(width, x1 + crop_width)
        y2 = min(height, y1 + crop_height)

        cropped = image[y1:y2, x1:x2]

        # 元サイズにリサイズ
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    def apply_hybrid(self, image, progress):
        # ズームとぼかしを組み合わせる
        # 例: ズームは線形に，ぼかしは後半早めに消えるように調整
        zoomed = self.apply_zoom(image, progress)

        # ぼかし用の進行度を少し早める (例: progress 0.8でぼかしゼロ)
        blur_progress = min(1.0, progress * 1.25)
        return self.apply_blur(zoomed, blur_progress)

    def resize_image(self, image, target_width, target_height):
        """
        画像をリサイズ

        Args:
            image: 入力画像
            target_width: 目標幅
            target_height: 目標高さ

        Returns:
            リサイズされた画像
        """
        if image is None:
            return None

        return cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )
