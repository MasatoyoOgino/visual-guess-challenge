"""
GameEngine - ゲームロジック管理クラス
モード管理、画像処理、スコア計算を担当
"""

import cv2
import os
from image_processor import ImageProcessor


class GameEngine:
    """ゲームエンジンクラス"""
    
    def __init__(self, image_path, mode='blur'):
        """
        初期化
        
        Args:
            image_path: 画像ファイルのパス
            mode: ゲームモード ('blur', 'zoom', 'hybrid')
        """
        self.image_path = image_path
        self.mode = mode
        self.original_image = None
        self.current_image = None
        self.current_level = 1
        self.max_level = 5
        self.correct_answer = ""
        
        # 画像プロセッサのインスタンス
        self.image_processor = ImageProcessor()
        
        # 画像の読み込み
        self.load_image()
        
        # 正解キーワードの抽出（ファイル名から）
        self.extract_answer_from_filename()
        
        # 初期画像の生成
        self.update_image()
        
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
        # 例: "cat_image.jpg" -> "cat"
        self.correct_answer = name_without_ext.split('_')[0].split('-')[0].lower()
        
    def set_answer(self, answer):
        """正解を手動で設定"""
        self.correct_answer = answer.lower()
    
    def update_image(self):
        """現在のレベルに応じて画像を更新"""
        if self.original_image is None:
            return
        
        if self.mode == 'blur':
            self.current_image = self.image_processor.apply_blur(
                self.original_image, self.current_level, self.max_level
            )
        elif self.mode == 'zoom':
            self.current_image = self.image_processor.apply_zoom(
                self.original_image, self.current_level, self.max_level
            )
        elif self.mode == 'hybrid':
            self.current_image = self.image_processor.apply_hybrid(
                self.original_image, self.current_level, self.max_level
            )
        else:
            self.current_image = self.original_image.copy()
    
    def get_current_image(self):
        """現在の画像を取得"""
        return self.current_image
    
    def get_current_level(self):
        """現在のレベルを取得"""
        return self.current_level
    
    def increase_level(self):
        """レベルを上げる"""
        if self.current_level < self.max_level:
            self.current_level += 1
            self.update_image()
    
    def decrease_level(self):
        """レベルを下げる"""
        if self.current_level > 1:
            self.current_level -= 1
            self.update_image()
    
    def set_level(self, level):
        """レベルを設定"""
        if 1 <= level <= self.max_level:
            self.current_level = level
            self.update_image()
    
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
        is_correct = (user_answer_lower == correct_answer_lower or 
                     correct_answer_lower in user_answer_lower or
                     user_answer_lower in correct_answer_lower)
        
        return is_correct, self.correct_answer
    
    def calculate_score(self, elapsed_seconds):
        """
        スコアを計算
        
        Args:
            elapsed_seconds: 経過時間（秒）
            
        Returns:
            スコア（0-100点）
        """
        # スコア計算式: score = max(0, 100 - 10*(level-1) - 2*elapsed_sec)
        score = max(0, 100 - 10 * (self.current_level - 1) - 2 * elapsed_seconds)
        return score
    
    def get_mode(self):
        """現在のモードを取得"""
        return self.mode
    
    def get_correct_answer(self):
        """正解を取得"""
        return self.correct_answer

