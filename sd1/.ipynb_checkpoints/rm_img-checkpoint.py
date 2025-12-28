import os

def delete_images_in_range(directory, start=30000, end=59999):
    for i in range(start, end + 1):
        filename = f"{i}.png"
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# 使用示例
directory = 'outputs/tome/coco30k/ratio_0.4/samples'  # 替换为你的目录路径
delete_images_in_range(directory)