import os
import numpy as np
import time
import sys

# --- 关键导入 ---
# 确保你已经 'pip install ultralytics'
# FastSAMPredictor 通常在这里：

from ultralytics.models.fastsam import FastSAMPredictor


def main():
    
    try:
        
        # __file__ 是当前脚本的路径
        script_dir = os.path.abspath(os.path.split(os.path.realpath(__file__))[0])
        # --- 重点：模型路径 ---
        model_path = os.path.join(os.path.dirname(script_dir), 'resources/models', "FastSAM-x.pt")        
        # --- 关键信息：你使用的是 'FastSAM-x.pt' (eXtra Large) ---
        # --- 这极有可能是导致 8GB Orin 内存溢出的原因 ---
        print("警告: 正在加载 'FastSAM-x.pt' (Extra Large) 模型。")

        overrides = dict(
            conf=0.4, 
            task="segment", 
            mode="predict", 
            model=model_path, 
            save=False, 
            imgsz=640
        )
        

        start_time = time.time()
        
        # 这是最可能触发错误的地方
        predictor = FastSAMPredictor(overrides=overrides) 
        
        load_time = time.time() - start_time
        print(f"...FastSAMPredictor loaded. (耗时: {load_time:.2f} 秒)")
        
        print("Warming up FastSAM model (执行第一次推理)...")
        start_time = time.time()
        
        # 这是第二个可能触发错误的地方
        predictor(np.zeros((640, 480, 3), dtype=np.uint8)) 
        
        warmup_time = time.time() - start_time
        print(f"...FastSAM model warmed up. (耗时: {warmup_time:.2f} 秒)")
        
        print("\n" + "="*30)
        print("--- ✅ 测试成功 ---")
        print("如果此脚本成功运行，说明模型本身和环境是兼容的。")
        print("问题在于 ROS2 节点或其他进程占用了过多内存，导致运行 ROS 节点时没有足够剩余内存给模型。")
        print("="*30)

    except RuntimeError as e:
        print("\n" + "="*30)
        print("--- ❌ 测试失败 (符合预期) ---")
        print("成功复现了错误！")
        print(f"捕获到的 Runtime Error: {e}")
        print("\n这 99% 确认是 Jetson Orin 内存不足。")
        print("原因: 'FastSAM-x.pt' 模型对于 8GB 共享内存来说太大了。")
        print("="*30)
    except Exception as e:
        print(f"\n--- ❌ 测试失败 (意外错误) ---")
        print(f"捕获到的意外错误: {e}")

if __name__ == "__main__":
    main()