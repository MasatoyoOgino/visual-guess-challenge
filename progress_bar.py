from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtCore import Qt

class ProgressBar(QProgressBar):
    """
    ゲームの進行状況（画像の鮮明度など）を表示するプログレスバー
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """UIの初期化"""
        self.setRange(0, 100)
        self.setValue(0)
        self.setTextVisible(True)
        self.setAlignment(Qt.AlignCenter)
        self.setFormat("鮮明度: %p%")
        
        # スタイルの設定
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                background-color: #ecf0f1;
                height: 25px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 10px;
                margin: 0.5px;
            }
        """)

    def update_progress(self, value):
        """
        進捗状況を更新する
        
        Args:
            value (float): 進捗値 (0.0 - 1.0) または パーセント (0 - 100)
        """
        # 0.0-1.0の範囲なら100倍する、それ以外はそのまま
        if 0.0 <= value <= 1.0:
            percentage = int(value * 100)
        else:
            percentage = int(value)
            
        self.setValue(percentage)
        
        # 進行度に応じて色を変えるなどの演出も可能
        if percentage >= 100:
            self.setStyleSheet(self.styleSheet().replace("#3498db", "#2ecc71")) # 緑色に変更
        else:
            self.setStyleSheet(self.styleSheet().replace("#2ecc71", "#3498db")) # 青色に戻す
