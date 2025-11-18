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
    
    def apply_blur(self, image, current_level, max_level):
        """
        ガウシアンボケを適用
        
        Args:
            image: 入力画像（RGB形式）
            current_level: 現在のレベル（1-max_level）
            max_level: 最大レベル
            
        Returns:
            処理された画像
        """
        if image is None:
            return None
        
        # レベルに応じてσ値を計算
        # レベル1: σ=15, レベル2: σ=8, レベル3: σ=4, レベル4: σ=2, レベル5: σ=0
        sigma_values = [15, 8, 4, 2, 0]
        
        if current_level <= len(sigma_values):
            sigma = sigma_values[current_level - 1]
        else:
            sigma = 0
        
        if sigma == 0:
            return image.copy()
        
        # ガウシアンフィルタを適用
        # kernel_sizeはσに基づいて計算（奇数にする）
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred
    
    def apply_zoom(self, image, current_level, max_level):
        """
        ズーム処理を適用（中央部分を切り出して拡大）
        
        Args:
            image: 入力画像（RGB形式）
            current_level: 現在のレベル（1-max_level）
            max_level: 最大レベル
            
        Returns:
            処理された画像
        """
        if image is None:
            return None
        
        height, width = image.shape[:2]
        
        # レベルに応じて表示領域の割合を計算
        # レベル1: 1/8, レベル2: 1/4, レベル3: 1/2, レベル4: 3/4, レベル5: 全体
        zoom_ratios = [1/8, 1/4, 1/2, 3/4, 1.0]
        
        if current_level <= len(zoom_ratios):
            ratio = zoom_ratios[current_level - 1]
        else:
            ratio = 1.0
        
        # 切り出す領域のサイズを計算
        crop_width = int(width * ratio)
        crop_height = int(height * ratio)
        
        # 中央を切り出す
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2
        
        cropped = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
        
        # 元のサイズにリサイズ
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def apply_hybrid(self, image, current_level, max_level):
        """
        ハイブリッド処理（ぼかし＋ズーム）を適用
        
        Args:
            image: 入力画像（RGB形式）
            current_level: 現在のレベル（1-max_level）
            max_level: 最大レベル
            
        Returns:
            処理された画像
        """
        if image is None:
            return None
        
        # まずズーム処理を適用
        zoomed = self.apply_zoom(image, current_level, max_level)
        
        # 次にぼかし処理を適用（レベルが低いほど強くぼかす）
        # ハイブリッドモードでは、ぼかしの強度を調整
        blur_level = max(1, current_level - 2)  # ズームよりぼかしを早く解消
        blurred = self.apply_blur(zoomed, blur_level, max_level)
        
        return blurred
    
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
        
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

