"""
Visual Guess Challenge - メインアプリケーション
PyQtを使用したGUIアプリケーションのメインウィンドウ
"""

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QMessageBox,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import os

from game_engine import GameEngine
from timer_controller import TimerController
from image_processor import ImageProcessor
from dataset_loader import DatasetLoader


class MainWindow(QMainWindow):
    """メインウィンドウクラス"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Guess Challenge")
        self.setGeometry(100, 100, 800, 600)

        # ゲーム関連のインスタンス
        self.game_engine = None
        self.timer_controller = TimerController()
        self.image_processor = ImageProcessor()
        self.dataset_loader = DatasetLoader()

        # UIコンポーネント
        self.mode_buttons = {}
        self.image_label = None
        self.answer_input = None
        self.submit_button = None
        self.time_label = None
        self.score_label = None
        self.hint_label = None
        self.message_label = None

        # タイマー
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)

        self.init_ui()

    def init_ui(self):
        """UIの初期化"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # モード選択エリア
        mode_layout = QHBoxLayout()
        mode_label = QLabel("モード選択：")
        self.mode_buttons["blur"] = QPushButton("Blur")
        self.mode_buttons["zoom"] = QPushButton("Zoom")
        self.mode_buttons["hybrid"] = QPushButton("Hybrid")

        for btn in self.mode_buttons.values():
            btn.setCheckable(True)
            btn.clicked.connect(self.on_mode_selected)

        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_buttons["blur"])
        mode_layout.addWidget(self.mode_buttons["zoom"])
        mode_layout.addWidget(self.mode_buttons["hybrid"])
        mode_layout.addStretch()

        # 操作ボタンエリア
        control_layout = QHBoxLayout()
        load_button = QPushButton("画像読み込み")
        random_button = QPushButton("ランダム画像")
        next_button = QPushButton("次へ")
        reset_button = QPushButton("リセット")

        load_button.clicked.connect(self.load_image)
        random_button.clicked.connect(self.load_random_image)
        next_button.clicked.connect(self.next_question)
        reset_button.clicked.connect(self.reset_game)

        control_layout.addWidget(load_button)
        control_layout.addWidget(random_button)
        control_layout.addWidget(next_button)
        control_layout.addWidget(reset_button)
        control_layout.addStretch()

        # 情報表示エリア
        info_layout = QHBoxLayout()
        self.time_label = QLabel("経過時間：00.0s")
        self.score_label = QLabel("スコア：---")
        info_layout.addWidget(self.time_label)
        info_layout.addWidget(self.score_label)
        info_layout.addStretch()

        # 画像表示エリア
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setStyleSheet(
            "border: 2px solid gray; background-color: #f0f0f0;"
        )
        self.image_label.setText("画像がここに表示されます")

        # 回答入力エリア
        answer_layout = QHBoxLayout()
        answer_label = QLabel("回答入力：")
        self.answer_input = QLineEdit()
        self.answer_input.setPlaceholderText("ここに回答を入力してください")
        self.submit_button = QPushButton("回答する")
        self.submit_button.clicked.connect(self.submit_answer)
        self.answer_input.returnPressed.connect(self.submit_answer)

        answer_layout.addWidget(answer_label)
        answer_layout.addWidget(self.answer_input)
        answer_layout.addWidget(self.submit_button)

        # ヒント・メッセージエリア
        hint_layout = QHBoxLayout()
        self.hint_label = QLabel("ヒントレベル：☆☆☆☆☆")
        self.message_label = QLabel("メッセージ：何が映っているでしょう？")
        hint_layout.addWidget(self.hint_label)
        hint_layout.addWidget(self.message_label)

        # レイアウトの組み立て
        main_layout.addLayout(mode_layout)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(info_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(answer_layout)
        main_layout.addLayout(hint_layout)

    def on_mode_selected(self):
        """モード選択時の処理"""
        sender = self.sender()
        # 他のボタンのチェックを外す
        for btn in self.mode_buttons.values():
            if btn != sender:
                btn.setChecked(False)

        # 選択されたモードを取得
        selected_mode = None
        for mode, btn in self.mode_buttons.items():
            if btn.isChecked():
                selected_mode = mode
                break

        if selected_mode:
            self.message_label.setText(
                f"メッセージ：{selected_mode.upper()}モードが選択されました"
            )

    def load_image(self):
        """画像を読み込む"""
        # デフォルトでimagesフォルダを開く
        default_dir = os.path.join(os.path.dirname(__file__), "images")
        if not os.path.exists(default_dir):
            default_dir = os.path.dirname(__file__)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "画像を選択",
            default_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)",
        )

        if file_path:
            # モードが選択されているか確認
            selected_mode = None
            for mode, btn in self.mode_buttons.items():
                if btn.isChecked():
                    selected_mode = mode
                    break

            if not selected_mode:
                QMessageBox.warning(self, "警告", "先にモードを選択してください")
                return

            # ゲームエンジンの初期化
            self.game_engine = GameEngine(file_path, selected_mode)
            self.timer_controller.start()
            self.update_display()
            self.update_timer.start(100)  # 100msごとに更新

    def update_display(self):
        """画面の更新"""
        if not self.game_engine:
            return

        # タイマー更新
        elapsed = self.timer_controller.get_elapsed_time()
        self.time_label.setText(f"経過時間：{elapsed:.1f}s")

        # 画像表示
        # 変更点: GameEngineに経過時間を渡して，その瞬間の画像を取得する
        processed_image = self.game_engine.get_processed_image(elapsed)
        if processed_image is not None:
            self.display_image(processed_image)

        # ヒント/進行度表示
        # 変更点: レベル(1-5)の概念がなくなったため，進行度を「鮮明度(%)」として表示
        time_limit = self.game_engine.time_limit
        if time_limit > 0:
            progress = min(1.0, elapsed / time_limit)
        else:
            progress = 1.0

        # 進行度をパーセント表示 (例: 鮮明度 0% -> 100%)
        clarity_percent = int(progress * 100)
        self.hint_label.setText(f"鮮明度：{clarity_percent}%")

    def display_image(self, image):
        """画像を表示する"""
        if image is None:
            return

        # 念のためメモリの連続性を確保（前回の安全策も含めて記述します）
        if not image.flags["C_CONTIGUOUS"]:
            image = image.copy()

        height, width, channel = image.shape
        bytes_per_line = 3 * width

        # 修正箇所: .rgbSwapped() を削除しました
        q_image = QImage(
            image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )

        # QPixmapに変換して表示
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.contentsRect().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def update_hint_display(self, level):
        """ヒントレベルの表示を更新"""
        max_level = 5
        stars = "★" * level + "☆" * (max_level - level)
        self.hint_label.setText(f"ヒントレベル：{stars}")

    def submit_answer(self):
        """回答を提出"""
        if not self.game_engine:
            QMessageBox.warning(self, "警告", "先に画像を読み込んでください")
            return

        answer = self.answer_input.text().strip()
        if not answer:
            QMessageBox.warning(self, "警告", "回答を入力してください")
            return

        # 正答判定
        is_correct, correct_answer = self.game_engine.check_answer(answer)
        elapsed = self.timer_controller.get_elapsed_time()

        if is_correct:
            score = self.game_engine.calculate_score(elapsed)
            self.score_label.setText(f"スコア：{score:.1f}点")
            self.message_label.setText("正解です！認識スピードは驚異的！")
            QMessageBox.information(
                self, "正解！", f"正解です！\nスコア：{score:.1f}点"
            )
        else:
            self.message_label.setText(f"残念！正解は「{correct_answer}」でした。")
            QMessageBox.warning(
                self, "不正解", f"残念！正解は「{correct_answer}」でした。"
            )

        # タイマー停止
        self.timer_controller.stop()
        self.update_timer.stop()

    def load_random_image(self):
        """ランダムに画像を読み込む"""
        # モードが選択されているか確認
        selected_mode = None
        for mode, btn in self.mode_buttons.items():
            if btn.isChecked():
                selected_mode = mode
                break

        if not selected_mode:
            QMessageBox.warning(self, "警告", "先にモードを選択してください")
            return

        # データセットからランダムに画像を選択
        image_path = self.dataset_loader.get_random_image()

        if image_path is None:
            QMessageBox.warning(
                self,
                "警告",
                "imagesフォルダに画像がありません。\n"
                "imagesフォルダに画像ファイルを配置してください。",
            )
            return

        # ゲームエンジンの初期化
        self.game_engine = GameEngine(image_path, selected_mode)
        self.timer_controller.start()
        self.update_display()
        self.update_timer.start(100)  # 100msごとに更新
        self.message_label.setText(
            f"メッセージ：ランダム画像を読み込みました（{os.path.basename(image_path)}）"
        )

    def next_question(self):
        """次の問題へ"""
        # ランダムに次の画像を読み込む
        if self.game_engine:
            # 現在のモードを保持
            current_mode = self.game_engine.get_mode()
            # モードボタンを更新
            for mode, btn in self.mode_buttons.items():
                btn.setChecked(mode == current_mode)

        # リセットしてからランダム画像を読み込む
        self.reset_game()
        # モードが選択されていれば自動的にランダム画像を読み込む
        selected_mode = None
        for mode, btn in self.mode_buttons.items():
            if btn.isChecked():
                selected_mode = mode
                break

        if selected_mode:
            self.load_random_image()

    def reset_game(self):
        """ゲームをリセット"""
        self.game_engine = None
        self.timer_controller.reset()
        self.update_timer.stop()
        self.image_label.clear()
        self.image_label.setText("画像がここに表示されます")
        self.answer_input.clear()
        self.time_label.setText("経過時間：00.0s")
        self.score_label.setText("スコア：---")
        self.hint_label.setText("ヒントレベル：☆☆☆☆☆")
        self.message_label.setText("メッセージ：何が映っているでしょう？")

        # モードボタンのチェックを外す
        for btn in self.mode_buttons.values():
            btn.setChecked(False)


def main():
    """メイン関数"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
