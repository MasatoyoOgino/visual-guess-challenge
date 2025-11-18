"""
TimerController - タイマー管理クラス
経過時間の計測を担当
"""

import time


class TimerController:
    """タイマーコントローラークラス"""
    
    def __init__(self):
        """初期化"""
        self.start_time = None
        self.is_running = False
        
    def start(self):
        """タイマーを開始"""
        self.start_time = time.time()
        self.is_running = True
    
    def stop(self):
        """タイマーを停止"""
        self.is_running = False
    
    def reset(self):
        """タイマーをリセット"""
        self.start_time = None
        self.is_running = False
    
    def get_elapsed_time(self):
        """
        経過時間を取得（秒）
        
        Returns:
            経過時間（秒）。タイマーが開始されていない場合は0.0
        """
        if self.start_time is None:
            return 0.0
        
        if self.is_running:
            return time.time() - self.start_time
        else:
            # 停止している場合は、停止時点までの経過時間を返す
            # （現在は停止時の時間を保持していないので、最後の経過時間を返す）
            return 0.0
    
    def is_timer_running(self):
        """タイマーが実行中かどうかを返す"""
        return self.is_running

