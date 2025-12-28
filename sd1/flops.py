from calflops import calculate_flops
from openaimodel import UNetModel
import torch

# 初始化模型
model = UNetModel(
    image_size=512,               # 输入图像的大小，例如64x64
    in_channels=4,               # 输入通道数，例如潜在空间的通道数
    model_channels=128,          # 模型的基通道数
    out_channels=4,              # 输出通道数
    num_res_blocks=2,            # 残差块的数量
    attention_resolutions=(16,), # 注意力分辨率
    dropout=0.1,                 # dropout概率
    channel_mult=(1, 2, 4, 8),   # 每层的通道倍增
    num_heads=4,                 # 注意力头数
)
dummy_input = torch.randn(1, 4, 512, 512)
timesteps = torch.randint(0, 1000, (1,))  # 随机时间步，用于模型的时序嵌入

# 使用calflops计算FLOPs
flops = calculate_flops(model, input_data=(dummy_input, timesteps))
print(f"FLOPs: {flops}")
