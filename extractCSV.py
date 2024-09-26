import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


def process_event_file(log_file_path, output_folder):
    # 创建EventAccumulator对象并加载数据
    event_acc = EventAccumulator(log_file_path)
    event_acc.Reload()

    # 获取所有可用的标签
    available_tags = event_acc.Tags()['scalars']

    # 准备数据字典
    data = {}
    steps = None

    for tag in available_tags:
        events = event_acc.Scalars(tag)
        if steps is None:
            steps = [event.step for event in events]
        values = [event.value for event in events]
        data[tag] = values

    data['Step'] = steps
    df = pd.DataFrame(data)

    # 提取文件夹名称作为CSV文件名
    folder_name = os.path.basename(os.path.dirname(log_file_path))
    csv_file_path = os.path.join(output_folder, f"{folder_name}.csv")

    # 保存为CSV文件
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")


def main(logs_folder):
    # 创建输出文件夹
    output_folder = os.path.join(logs_folder, 'csv')
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(logs_folder):
        for file_name in files:
            if file_name.startswith('events.out.tfevents'):
                log_file_path = os.path.join(root, file_name)
                process_event_file(log_file_path, output_folder)


if __name__ == "__main__":
    logs_folder = 'logs'
    main(logs_folder)
