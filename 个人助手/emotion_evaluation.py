from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, LSTM, Bidirectional, Lambda, Dropout, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Layer
import numpy as np
import cv2
from typing import Dict, List, Tuple
import os
import tensorflow as tf
import glob
from datetime import datetime
import threading
import time

class CNNFeatureExtractor(Layer):
    """CNN特征提取器层"""
    def __init__(self, **kwargs):
        super(CNNFeatureExtractor, self).__init__(**kwargs)
        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.pool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pool2 = MaxPooling2D((2, 2))
        self.conv3 = Conv2D(128, (3, 3), activation='relu')
        self.pool3 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense = Dense(128, activation='relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 128)

class EmotionEvaluator:
    def __init__(self, model_path='emotion_model.keras'):
        """初始化情绪评估器"""
        self.model = self._load_or_create_model(model_path)
        self.emotions = ['愤怒', '厌恶', '恐惧', '快乐', '悲伤', '惊讶', '平静']
        self.is_evaluating = False
        self.evaluation_thread = None
        self.last_evaluation_time = 0
        self.evaluation_interval = 5  # 每5秒评估一次
        
    def _load_or_create_model(self, model_path):
        """加载或创建模型"""
        if os.path.exists(model_path):
            try:
                print("正在加载模型...")
                model = load_model(model_path, safe_mode=False,
                                 custom_objects={'CNNFeatureExtractor': CNNFeatureExtractor})
                print("模型加载成功")
                return model
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
        
        print("创建新模型...")
        model = create_emotion_model()
        model.save(model_path)
        return model
    
    def start_evaluation(self):
        """开始自动评估"""
        if not self.is_evaluating:
            self.is_evaluating = True
            self.evaluation_thread = threading.Thread(target=self._evaluation_loop)
            self.evaluation_thread.daemon = True
            self.evaluation_thread.start()
            print("开始自动情绪评估...")
    
    def stop_evaluation(self):
        """停止自动评估"""
        self.is_evaluating = False
        if self.evaluation_thread and self.evaluation_thread.is_alive():
            self.evaluation_thread.join(timeout=1)
        print("停止自动情绪评估")
    
    def _evaluation_loop(self):
        """评估循环"""
        while self.is_evaluating:
            current_time = time.time()
            if current_time - self.last_evaluation_time >= self.evaluation_interval:
                self._evaluate_latest_video()
                self.last_evaluation_time = current_time
            time.sleep(0.1)  # 短暂休眠以减少CPU使用
    
    def _evaluate_latest_video(self):
        """评估最新的视频文件"""
        try:
            # 查找最新的视频文件
            video_files = glob.glob('video_*.avi')
            if not video_files:
                return
            
            # 按修改时间排序
            latest_video = max(video_files, key=os.path.getmtime)
            
            # 分析情绪
            emotion_scores = self.analyze_emotions(latest_video, self.model)
            
            # 保存分析结果
            save_analysis_results(latest_video, emotion_scores)
            
            # 打印结果
            print(f"\n视频 {latest_video} 的情绪分析结果:")
            print("=" * 50)
            for emotion, score in emotion_scores:
                print(f"{emotion}: {score:.4f}")
            print("=" * 50)
            
        except Exception as e:
            print(f"评估视频时发生错误: {str(e)}")
    
    def analyze_emotions(self, file_path: str, model) -> List[Tuple[str, float]]:
        """分析视频中的情绪"""
        try:
            # 预处理视频
            preprocessed_video = preprocess_video(file_path)
            preprocessed_video = np.expand_dims(preprocessed_video, axis=0)
            
            # 预测情绪
            predictions = model.predict(preprocessed_video)
            
            # 获取每个情绪的置信度
            emotion_scores = list(zip(self.emotions, predictions[0]))
            
            # 按置信度排序
            emotion_scores.sort(key=lambda x: x[1], reverse=True)
            
            return emotion_scores
        except Exception as e:
            raise ValueError(f"情绪分析失败: {str(e)}")

def create_emotion_model():
    """创建情绪分析模型"""
    # 视频输入分支
    visual_input = Input(shape=(6, 128, 128, 3))
    
    # 使用TimeDistributed包装CNN特征提取器
    cnn_layer = CNNFeatureExtractor()
    encoded_frames = TimeDistributed(cnn_layer)(visual_input)
    
    # 使用LSTM处理时序特征
    encoded_vid = LSTM(64, return_sequences=False)(encoded_frames)
    
    # 添加全连接层
    x = Dense(128, activation='relu')(encoded_vid)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # 输出层 - 7种基本情绪
    outputs = Dense(7, activation='softmax')(x)
    
    # 创建模型
    model = Model(inputs=visual_input, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def get_number_of_frames(file_path: str) -> int:
    """获取视频总帧数"""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {file_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def extract_N_video_frames(file_path: str, number_of_samples: int = 6) -> list:
    """提取视频帧"""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {file_path}")
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"视频文件为空: {file_path}")
    
    # 计算采样间隔
    indices = np.linspace(0, total_frames-1, number_of_samples, dtype=int)
    
    video_frames = []
    for ind in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ind)
        ret, frame = cap.read()
        if ret:
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame_rgb)
    
    cap.release()
    
    if len(video_frames) == 0:
        raise ValueError(f"无法从视频中提取帧: {file_path}")
    
    return video_frames

def resize_image(image: np.ndarray, new_size: tuple) -> np.ndarray:
    """调整图像大小"""
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def preprocess_video(file_path: str) -> np.ndarray:
    """预处理视频帧"""
    try:
        # 提取视频帧
        sampled = extract_N_video_frames(file_path=file_path, number_of_samples=6)
        # 调整大小
        resized_images = [resize_image(image=im, new_size=(128, 128)) for im in sampled]
        # 归一化
        preprocessed_video = np.stack(resized_images) / 255.0
        return preprocessed_video
    except Exception as e:
        raise ValueError(f"视频预处理失败: {str(e)}")

def save_analysis_results(file_path: str, emotion_scores: List[Tuple[str, float]]):
    """保存分析结果到文件"""
    try:
        # 创建results目录（如果不存在）
        os.makedirs('results', exist_ok=True)
        
        # 使用视频文件名作为基础
        video_name = os.path.splitext(os.path.basename(file_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join('results', f'emotion_analysis_{video_name}_{timestamp}.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("情绪分析结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"视频文件: {file_path}\n")
            f.write(f"分析时间: {timestamp}\n")
            f.write("-" * 50 + "\n")
            f.write("情绪得分:\n")
            for emotion, score in emotion_scores:
                f.write(f"{emotion}: {score:.4f}\n")
            f.write("=" * 50 + "\n")
        
        print(f"分析结果已保存到: {output_file}")
        return output_file
    except Exception as e:
        print(f"保存分析结果失败: {str(e)}")
        return None

if __name__ == "__main__":
    # 创建情绪评估器实例
    evaluator = EmotionEvaluator()
    
    try:
        # 启动自动评估
        evaluator.start_evaluation()
        
        # 保持程序运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n程序终止")
    finally:
        # 停止评估
        evaluator.stop_evaluation()