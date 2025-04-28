import os
import threading
import queue
import time
import asyncio
import edge_tts
from datetime import datetime
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.utilities import ArxivAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
import requests
import socket
import pyautogui
import base64
from PIL import Image
import io
from io import BytesIO
from langchain.callbacks import StdOutCallbackHandler
import cv2
import numpy as np

class VoiceChatBot:
    def __init__(self):
        """初始化语音聊天机器人"""
        # 加载环境变量
        load_dotenv()
        
        # 获取豆包API配置
        self.api_key = os.getenv("DOUBAO_API_KEY")
        self.api_base = os.getenv("DOUBAO_API_BASE")
        self.model_name = os.getenv("DOUBAO_MODEL_NAME")
        
        if not self.api_key or not self.api_base:
            raise ValueError("请在.env文件中设置DOUBAO_API_KEY和DOUBAO_API_BASE")
        
        # 检查网络连接
        self.is_online = self._check_network_connection()
        if not self.is_online:
            print("警告: 网络连接不可用，部分功能可能受限")
        
        # 初始化搜索工具
        self.tools = []
        if self.is_online:
            self._initialize_search_tools()
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            streaming=True,
            temperature=0.7,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base
        )
        
        # 初始化记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 初始化系统提示
        self.system_message = """你是一个智能助手，具有以下能力：
1. 可以通过网络搜索获取最新信息
2. 可以查询维基百科获取知识
3. 可以搜索学术论文
4. 当用户提出需要分析屏幕内容时，使用截图工具
5. 当用户提出需要分析摄像头画面内容时，使用摄像头分析工具
6. 当用户上传图片时，使用图片分析工具
7. 在回答问题时，你应该：
   - 如果用户需要分析屏幕内容，使用截图工具。即当用户没有开启摄像头也没有上传文件时，要分析画面或图片时，使用截图工具
   - 如果用户需要分析摄像头画面，使用摄像头分析工具。即当用户开启摄像头时，并且重点在'镜头'、'摄像头'、'我'、'我周围'时，使用摄像头分析工具
   - 如果用户上传图片，或者文件中包含图片时，使用图片分析工具。
   - 以上三种情况，你都需要先分析图片，结合用户的提示词，生成回答，注意要选择最合适的工具，不要混用工具
   - 首先使用搜索工具获取相关信息
   - 然后基于获取的信息和你的知识来回答问题
   - 如果搜索结果不相关，则使用你的知识回答
   - 在回答中引用信息来源
   - 如果搜索工具不可用，直接使用你的知识回答"""
        
        # 初始化Agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            agent_kwargs={
                "system_message": self.system_message,
                "max_iterations": 3
            }
        )
        
        # 初始化其他组件
        self._initialize_voice_components()
        
        # 添加搜索限制
        self.last_search_times = {}
        self.min_search_intervals = {
            "Wikipedia": 2.0,
            "Arxiv": 3.0
        }
        
        # 添加摄像头相关属性
        self.camera = None
        self.is_camera_active = False
        self.camera_lock = threading.Lock()
    
    def _initialize_search_tools(self):
        """初始化搜索工具"""
        try:
            # 初始化 Wikipedia
            self.wikipedia = WikipediaAPIWrapper(
                top_k_results=3,
                lang="zh"
            )
            self.tools.append(Tool(
                name="Wikipedia",
                func=self._wikipedia_search,
                description="用于在维基百科上搜索信息和资讯。输入应该是一个搜索查询。当用户询问关于历史、地理、科学、技术、文化、艺术、人物、事件等方面的问题时，可以使用此工具。"
            ))
            print("Wikipedia 搜索工具初始化成功")
        except Exception as e:
            print(f"Wikipedia 初始化失败: {str(e)}")
        
        try:
            # 初始化 Arxiv
            self.arxiv = ArxivAPIWrapper(
                top_k_results=3,
                load_all_available_meta=True,
                sort_by="relevance"
            )
            self.tools.append(Tool(
                name="Arxiv",
                func=self._arxiv_search,
                description="用于搜索学术论文。输入应该是一个搜索查询。当用户询问关于学术论文、研究成果、科学发现等方面的问题时，可以使用此工具。"
            ))
            print("Arxiv 搜索工具初始化成功")
        except Exception as e:
            print(f"Arxiv 初始化失败: {str(e)}")
        
        # 添加屏幕分析工具
        self.tools.append(
            Tool(
                name="ScreenAnalysis",
                func=self._capture_and_analyze_screen,
                description="用于分析屏幕内容。即当用户没有开启摄像头也没有上传文件而要分析屏幕内容、屏幕画面或屏幕图片时，使用截图工具"
            )
        )
        
        # 添加摄像头分析工具
        self.tools.append(
            Tool(
                name="CameraAnalysis",
                func=self._capture_and_analyze_camera,
                description="用于分析摄像头画面内容。当用户要求分析摄像头画面、镜头内容或视频画面、或问题主体为'我'、'我周围'时使用此工具。"
            )
        )
        
        if not self.tools:
            print("所有搜索工具初始化失败")

    def _initialize_voice_components(self):
        """初始化语音相关组件"""
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000  # 调整能量阈值
        self.recognizer.dynamic_energy_threshold = True  # 动态能量阈值
        self.recognizer.pause_threshold = 0.8  # 调整停顿阈值
        
        # 初始化语音合成设置
        self.tts_voice = "zh-CN-XiaoxiaoNeural"  # 默认使用晓晓声音
        self.tts_rate = "+0%"  # 正常语速
        self.tts_volume = "+0%"  # 正常音量
        self.tts_enabled = True  # 默认开启语音播报
        
        # 初始化语音引擎锁
        self.speech_lock = threading.Lock()
        
        # 初始化语音广播队列
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.current_speech_thread = None
        
        # 初始化麦克风
        self.mic = None
        self.initialize_microphone()
        
        # 添加状态控制
        self.is_processing = False
        self.current_response = ""
        
        # 添加语音监听控制
        self.is_listening = False
        self.listen_thread = None
        self.listen_callback = None
        
        # 添加语音播报控制
        self.should_stop_speaking = False
        self.speech_event = threading.Event()
        self.speech_event.set()  # 初始状态为已设置
        self.pending_speech = None  # 存储待播放的语音内容

    def initialize_microphone(self):
        try:
            # 获取可用的麦克风列表
            mic_list = sr.Microphone.list_microphone_names()
            if not mic_list:
                raise Exception("未找到可用的麦克风设备")
            
            # 尝试使用默认麦克风
            self.mic = sr.Microphone()
        except Exception as e:
            print(f"麦克风初始化错误: {str(e)}")
            raise

    def toggle_tts(self):
        """切换语音播报开关"""
        self.tts_enabled = not self.tts_enabled
        if not self.tts_enabled and self.is_speaking:
            self.stop_processing()
        return self.tts_enabled
        
    def start_listening(self, callback=None):
        """开始持续监听语音输入"""
        if not self.is_listening:
            self.is_listening = True
            self.listen_callback = callback
            self.listen_thread = threading.Thread(target=self._continuous_listen)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            print("已启动语音监听")
        
    def stop_listening(self):
        """停止持续监听"""
        self.is_listening = False
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=0.1)
        self.listen_thread = None
        self.listen_callback = None
        
    def _continuous_listen(self):
        """持续监听语音输入"""
        while self.is_listening:
            try:
                # 调整麦克风灵敏度
                with self.mic as source:
                    print("正在调整环境噪音...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    print("开始监听...")
                    # 增加超时时间，减少误报
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=15)
                    
                try:
                    print("正在识别语音...")
                    text = self.recognizer.recognize_google(audio, language='zh-CN')
                    print(f"识别结果: {text}")
                    
                    if text.strip():
                        # 使用回调函数处理识别结果
                        if self.listen_callback:
                            self.listen_callback(text)
                            
                except sr.UnknownValueError:
                    print("未能识别语音")
                except sr.RequestError as e:
                    print(f"语音识别服务出错: {e}")
                    time.sleep(1)  # 服务出错时等待较长时间
                    
            except sr.WaitTimeoutError:
                # 超时时不打印错误，继续监听
                continue
            except Exception as e:
                print(f"语音监听出错: {e}")
                time.sleep(0.1)  # 出错时短暂等待
                
    def _handle_voice_input(self, text):
        """处理语音输入"""
        if text.strip():
            # 如果正在处理或说话，立即停止当前的处理
            if self.is_processing or self.is_speaking:
                self.stop_processing()
            
            # 处理新的输入
            self.text_chat(text, self._display_message)
            
    def stop_processing(self):
        """停止当前的处理和语音播报"""
        with self.speech_lock:
            self.is_speaking = False
            self.is_processing = False
            self.should_stop_speaking = True
            self.speech_event.clear()  # 清除事件，表示需要停止
            
            # 清空语音队列
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
            
            # 等待当前语音线程结束
            if self.current_speech_thread and self.current_speech_thread.is_alive():
                self.current_speech_thread.join(timeout=0.1)  # 减少等待时间
                if self.current_speech_thread.is_alive():
                    # 如果线程还在运行，强制终止
                    self.current_speech_thread._stop()
            
            # 重置状态
            self.current_response = ""
            self.current_speech_thread = None
            self.should_stop_speaking = False
            self.speech_event.set()  # 重新设置事件，准备下一轮播报

    def text_chat(self, text, callback=None):
        """文本对话"""
        if self.is_processing:
            return
            
        self.is_processing = True
        try:
            # 如果正在说话，立即停止
            if self.is_speaking:
                self.stop_processing()
                time.sleep(0.1)  # 短暂等待确保停止完成
            
            # 使用 invoke 替代 run 方法
            response = self.agent.invoke({"input": text})
            
            # 正确处理响应
            if isinstance(response, dict):
                response_text = response.get("output", "抱歉，无法获取响应")
            else:
                response_text = str(response)
            
            # 显示响应
            if callback:
                callback(response_text)
                
            # 进行语音播报
            if self.tts_enabled:
                # 直接设置待播放内容
                self.pending_speech = response_text
                # 启动新的语音线程
                if not self.current_speech_thread or not self.current_speech_thread.is_alive():
                    self.current_speech_thread = threading.Thread(target=self._process_speech_queue)
                    self.current_speech_thread.daemon = True
                    self.current_speech_thread.start()
                
            return response_text
        except Exception as e:
            error_msg = f"处理请求时发生错误: {str(e)}"
            if callback:
                callback(error_msg)
            return error_msg
        finally:
            self.is_processing = False
    
    def speech_to_text(self):
        if not self.mic:
            return "麦克风未正确初始化"
            
        try:
            with self.mic as source:
                print("正在调整环境噪音...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("请开始说话...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("正在识别...")
                text = self.recognizer.recognize_google(audio, language="zh-CN")
                return text
        except sr.WaitTimeoutError:
            return "未检测到语音输入"
        except sr.UnknownValueError:
            return "无法识别语音内容"
        except sr.RequestError as e:
            return f"语音服务请求错误: {str(e)}"
        except Exception as e:
            return f"语音识别错误: {str(e)}"

    async def _do_tts(self, text, output_file):
        """执行文本转语音"""
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.tts_voice,
            rate=self.tts_rate,
            volume=self.tts_volume
        )
        await communicate.save(output_file)

    def text_to_speech(self, text):
        """将文本转换为语音"""
        if not self.tts_enabled or self.should_stop_speaking:
            return

        try:
            # 创建临时音频文件
            temp_file = "temp_speech.mp3"
            
            # 运行异步TTS
            asyncio.run(self._do_tts(text, temp_file))
            
            # 检查是否需要停止
            if self.should_stop_speaking:
                return
                
            # 使用 playsound 播放音频
            from playsound import playsound
            playsound(temp_file)
            
            # 删除临时文件
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            print(f"语音合成错误: {str(e)}")
        finally:
            self.speech_event.set()  # 确保事件被设置

    def _process_speech_queue(self):
        """处理语音队列"""
        while True:
            try:
                # 等待事件被设置
                self.speech_event.wait()
                
                # 检查是否有待播放的内容
                if self.pending_speech:
                    text = self.pending_speech
                    self.pending_speech = None  # 清除待播放内容
                    
                    # 检查是否需要停止
                    if self.should_stop_speaking:
                        continue
                    
                    with self.speech_lock:
                        self.is_speaking = True
                        self.text_to_speech(text)
                        self.is_speaking = False
                
                # 如果没有待播放内容，退出循环
                if not self.pending_speech:
                    break
                
            except Exception as e:
                print(f"语音处理错误: {str(e)}")
                self.is_speaking = False
                break

    def _check_network_connection(self):
        """检查网络连接状态"""
        try:
            # 尝试连接到一个可靠的网站
            requests.get("https://www.baidu.com", timeout=3)
            return True
        except requests.RequestException:
            try:
                # 如果百度连接失败，尝试连接Google
                requests.get("https://www.google.com", timeout=3)
                return True
            except requests.RequestException:
                return False

    def _wikipedia_search(self, query):
        """优化的 Wikipedia 搜索方法"""
        try:
            # 等待搜索间隔
            self._wait_for_rate_limit("Wikipedia")
            
            # 添加重试机制
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    result = self.wikipedia.run(query)
                    if result and len(result.strip()) > 0:
                        return result
                    return "未找到相关信息"
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    raise e
        except Exception as e:
            print(f"Wikipedia 搜索错误: {str(e)}")
            return "Wikipedia 搜索暂时不可用"

    def _arxiv_search(self, query):
        """优化的 Arxiv 搜索方法"""
        try:
            # 等待搜索间隔
            self._wait_for_rate_limit("Arxiv")
            
            # 添加重试机制
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    result = self.arxiv.run(query)
                    if result and len(result.strip()) > 0:
                        return result
                    return "未找到相关信息"
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    raise e
        except Exception as e:
            print(f"Arxiv 搜索错误: {str(e)}")
            return "Arxiv 搜索暂时不可用"

    def _capture_and_analyze_screen(self, query="请分析屏幕内容"):
        """捕获并分析屏幕内容"""
        try:
            # 捕获屏幕
            screenshot = pyautogui.screenshot()
            
            # 将图片转换为base64
            buffered = BytesIO()
            screenshot.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # 调用豆包视觉API
            return self._call_doubao_vision_api(img_str, query)
            
        except Exception as e:
            print(f"屏幕分析错误: {str(e)}")
            return "无法分析屏幕内容"

    def _call_doubao_vision_api(self, image_base64, query):
        """调用豆包视觉API"""
        try:
            api_url = f"{self.api_base}/chat/completions"  # 修改为正确的API端点
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": query
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            # 从响应中提取内容
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return "无法分析图片内容"
            
        except Exception as e:
            print(f"豆包API调用错误: {str(e)}")
            return "视觉分析服务暂时不可用"

    def _wait_for_rate_limit(self, tool_name):
        """等待搜索间隔"""
        current_time = time.time()
        elapsed = current_time - self.last_search_times.get(tool_name, 0)
        if elapsed < self.min_search_intervals.get(tool_name, 0):
            wait_time = self.min_search_intervals[tool_name] - elapsed
            time.sleep(wait_time)
        self.last_search_times[tool_name] = time.time()

    def toggle_camera(self):
        """切换摄像头状态"""
        with self.camera_lock:
            if not self.is_camera_active:
                try:
                    self.camera = cv2.VideoCapture(0)  # 打开默认摄像头
                    if not self.camera.isOpened():
                        raise Exception("无法打开摄像头")
                    self.is_camera_active = True
                    return True
                except Exception as e:
                    print(f"摄像头初始化错误: {str(e)}")
                    return False
            else:
                self.camera.release()
                self.camera = None
                self.is_camera_active = False
                return False

    def _capture_and_analyze_camera(self, query="请分析摄像头画面内容"):
        """捕获并分析摄像头画面"""
        if not self.is_camera_active:
            return "摄像头未开启，请先开启摄像头"
            
        try:
            with self.camera_lock:
                if not self.camera or not self.camera.isOpened():
                    return "摄像头未正确初始化"
                
                # 读取摄像头帧
                ret, frame = self.camera.read()
                if not ret:
                    return "无法获取摄像头画面"
                
                # 将图片转换为base64
                _, buffer = cv2.imencode('.png', frame)
                img_str = base64.b64encode(buffer).decode()
                
                # 调用豆包视觉API
                return self._call_doubao_vision_api(img_str, query)
                
        except Exception as e:
            print(f"摄像头分析错误: {str(e)}")
            return "无法分析摄像头画面"

    def __del__(self):
        """清理资源"""
        if self.camera:
            self.camera.release()