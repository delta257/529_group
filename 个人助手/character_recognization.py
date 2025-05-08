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

class CharacterFeatureExtractor(Layer):
    """特征提取器层"""
    def __init__(self, **kwargs):
        super(CharacterFeatureExtractor, self).__init__(**kwargs)
        self.conv1 = Conv2D(32, (3, 3), activation='relu', name='conv1')
        self.pool1 = MaxPooling2D((2, 2), name='pool1')
        self.conv2 = Conv2D(64, (3, 3), activation='relu', name='conv2')
        self.pool2 = MaxPooling2D((2, 2), name='pool2')
        self.conv3 = Conv2D(128, (3, 3), activation='relu', name='conv3')
        self.pool3 = MaxPooling2D((2, 2), name='pool3')
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(256, activation='relu', name='dense')

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
        return (input_shape[0], 256)

class CharacterAnalyzer:
    def __init__(self, model_path='character_model.keras'):
        """初始化性格分析器"""
        self.model = self._load_or_create_model(model_path)
        self.traits = [
            'neuroticism', 'extraversion', 'agreeableness', 'conscientiousness', 'openness'
        ]
        self.is_analyzing = False
        self.analysis_thread = None
        self.last_analysis_time = 0
        self.analysis_interval = 5  # 每5秒分析一次
        
    def _load_or_create_model(self, model_path):
        """加载或创建模型"""
        if os.path.exists(model_path):
            try:
                print("正在加载模型...")
                model = load_model(model_path, safe_mode=False,
                                 custom_objects={'CharacterFeatureExtractor': CharacterFeatureExtractor})
                print("模型加载成功")
                return model
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
        
        print("创建新模型...")
        model = create_character_model()
        model.save(model_path)
        return model
    
    def start_analysis(self):
        """开始自动分析"""
        if not self.is_analyzing:
            self.is_analyzing = True
            self.analysis_thread = threading.Thread(target=self._analysis_loop)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            print("开始自动性格分析...")
    
    def stop_analysis(self):
        """停止自动分析"""
        self.is_analyzing = False
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1)
        print("停止自动性格分析")
    
    def _analysis_loop(self):
        """分析循环"""
        while self.is_analyzing:
            current_time = time.time()
            if current_time - self.last_analysis_time >= self.analysis_interval:
                self._analyze_latest_video()
                self.last_analysis_time = current_time
            time.sleep(0.1)
    
    def _analyze_latest_video(self):
        """分析最新的视频文件"""
        try:
            # 查找最新的视频文件
            video_files = glob.glob('video_*.avi')
            if not video_files:
                return
            
            # 按修改时间排序
            latest_video = max(video_files, key=os.path.getmtime)
            
            # 分析性格特征
            trait_scores = self.analyze_character(latest_video, self.model)
            
            # 保存分析结果
            save_analysis_results(latest_video, trait_scores)
            
            # 打印结果
            print(f"\n视频 {latest_video} 的性格分析结果:")
            print("=" * 50)
            for trait, score in trait_scores:
                print(f"{trait}: {score:.4f}")
            print("=" * 50)
            
        except Exception as e:
            print(f"分析视频时发生错误: {str(e)}")
    
    def analyze_character(self, file_path: str, model) -> List[Tuple[str, float]]:
        """分析视频中的性格特征"""
        try:
            # 预处理视频
            preprocessed_video = preprocess_video(file_path)
            preprocessed_video = np.expand_dims(preprocessed_video, axis=0)
            
            # 预测性格特征
            predictions = model.predict(preprocessed_video)
            
            # 获取每个特征的得分
            trait_scores = list(zip(self.traits, predictions[0]))
            
            # 按得分排序
            trait_scores.sort(key=lambda x: x[1], reverse=True)
            
            return trait_scores
        except Exception as e:
            raise ValueError(f"性格分析失败: {str(e)}")

def create_character_model():
    """创建性格分析模型"""
    # 视频输入分支
    visual_input = Input(shape=(6, 128, 128, 3))
    
    # 使用TimeDistributed包装特征提取器
    feature_extractor = CharacterFeatureExtractor()
    encoded_frames = TimeDistributed(feature_extractor)(visual_input)
    
    # 使用LSTM处理时序特征
    encoded_vid = LSTM(128, return_sequences=False)(encoded_frames)
    
    # 添加全连接层
    x = Dense(256, activation='relu')(encoded_vid)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # 输出层 - 5个性格特征
    outputs = Dense(5, activation='sigmoid')(x)
    
    # 创建模型
    model = Model(inputs=visual_input, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
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

def save_analysis_results(file_path: str, trait_scores: List[Tuple[str, float]]):
    """保存分析结果到文件"""
    try:
        # 创建results目录（如果不存在）
        os.makedirs('results', exist_ok=True)
        
        # 使用视频文件名作为基础
        video_name = os.path.splitext(os.path.basename(file_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join('results', f'character_analysis_{video_name}_{timestamp}.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("性格分析结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"视频文件: {file_path}\n")
            f.write(f"分析时间: {timestamp}\n")
            f.write("-" * 50 + "\n")
            f.write("性格特征得分:\n")
            for trait, score in trait_scores:
                f.write(f"{trait}: {score:.4f}\n")
            f.write("=" * 50 + "\n")
        
        print(f"分析结果已保存到: {output_file}")
        return output_file
    except Exception as e:
        print(f"保存分析结果失败: {str(e)}")
        return None

if __name__ == "__main__":
    # 创建性格分析器实例
    analyzer = CharacterAnalyzer()
    
    try:
        # 启动自动分析
        analyzer.start_analysis()
        
        # 保持程序运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n程序终止")
    finally:
        # 停止分析
        analyzer.stop_analysis()