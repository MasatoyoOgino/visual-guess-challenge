"""
GameEngine - ゲームロジック管理クラス
時間経過に応じた画像処理とスコア計算，およびOpenCVによる自動クロップを担当
"""

import cv2
import os
import json  # 追加: JSON読み込み用
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
        self.correct_answer_key = ""  # 変数名を変更 (keyとして扱うため)
        self.synonyms = {}  # 追加: 類義語辞書

        self.image_processor = ImageProcessor()
        
        # 追加: 類義語辞書の読み込み
        self.load_synonyms()

        # 1. 正解キーワードの抽出
        self.extract_answer_from_filename()

        # 2. 画像の読み込み
        self.load_image()

        # 3. OpenCVによる主要被写体の検出とクロップ
        self.crop_to_main_subject()

    def load_synonyms(self):  # 追加: 辞書読み込みメソッド
        """類義語辞書(synonyms.json)を読み込む"""
        json_path = os.path.join(os.path.dirname(__file__), "synonyms.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.synonyms = json.load(f)
            except Exception as e:
                print(f"辞書ファイルの読み込みエラー: {e}")
                self.synonyms = {}
        else:
            self.synonyms = {}

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
        # キーワードを取得（例: dog_1.jpg -> dog）
        self.correct_answer_key = name_without_ext.split("_")[0].split("-")[0].lower()

    # (crop_to_main_subject, set_answer, get_processed_image は変更なしのため省略)
    # ... (既存のコードを維持)

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
        self.correct_answer_key = answer.lower()

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
        user_answer_norm = user_answer.strip().lower()
        key = self.correct_answer_key
        
        # 有効な正解リストを作成
        valid_answers = set()
        valid_answers.add(key)  # ファイル名そのものも正解に含める

        # JSON辞書にキーがあれば、そのリストを追加
        if key in self.synonyms:
            # リスト内の単語も小文字化・空白除去してセットに追加
            for synonym in self.synonyms[key]:
                valid_answers.add(str(synonym).strip().lower())
        
        # 判定: ユーザの回答が正解セットに含まれているか
        is_correct = user_answer_norm in valid_answers
        
        # 元のロジック（部分一致）を残したい場合は以下のように条件を追加可能ですが、
        # "犬" などの短い単語での誤爆を防ぐため、上記のような完全一致(in set)が推奨されます。
        
        # 表示用の正解文字列（代表値）
        display_answer = self.synonyms[key][1] if (key in self.synonyms and len(self.synonyms[key]) > 1) else key

        return is_correct, display_answer

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
        # 辞書にあれば日本語（リストの2番目と仮定）を返す、なければキーを返すなどの工夫が可能
        if self.correct_answer_key in self.synonyms and len(self.synonyms[self.correct_answer_key]) > 1:
             # 例としてリストの2番目の要素（日本語表記など）を正解として表示
             return self.synonyms[self.correct_answer_key][1]
        return self.correct_answer_key