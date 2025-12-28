import os

def count_photos_in_directory(directory):
    # 定义常见的图片文件扩展名
    photo_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    # 初始化计数器
    photo_count = 0
    
    # 遍历目录中的文件
    for filename in os.listdir(directory):
        # 获取文件的扩展名
        _, ext = os.path.splitext(filename)
        # 检查扩展名是否在定义的图片扩展名集合中
        if ext.lower() in photo_extensions:
            photo_count += 1
            
    return photo_count

x_values=[0.3,0.4,0.5,0.6,0.7]
for x in x_values:
    # 使用示例
    directory_path = f'outputs/sito/coco30k/localrandom/ratio_{x}_localrandom/samples'  # 替换为你的文件夹路径
    photo_count = count_photos_in_directory(directory_path)
    print(f"该文件夹下的照片数量: {photo_count}")