import os
import sys
from dotenv import load_dotenv
from VoiceChatBot import VoiceChatBot
from ChatGUI import ChatGUI

def main():
    # 加载环境变量
    load_dotenv()
    
    # 创建机器人实例
    bot = VoiceChatBot()
    
    # 创建并运行GUI
    app = ChatGUI(bot)
    app.mainloop()

if __name__ == "__main__":
    main()