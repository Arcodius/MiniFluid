from collections import defaultdict
import re


def calculate_profile_averages(data_source):
  # 用于存储每个函数的耗时列表
  times_map = defaultdict(list)

  # 正则表达式匹配格式：[Profile]: <func_name> | Time: <time_val> μs
  pattern = re.compile(r'\[Profile\]:\s*(.*?)\s*\|\s*Time:\s*(\d+)\s*μs')

  # 判断输入是文件路径还是纯文本
  if '\n' in data_source or len(data_source.splitlines()) > 1:
    lines = data_source.strip().splitlines()
  else:
    with open(data_source, 'r', encoding='utf-8') as f:
      lines = f.readlines()

  for line in lines:
    match = pattern.search(line)
    if match:
      func_name = match.group(1).strip()
      time_val = int(match.group(2))
      times_map[func_name].append(time_val)

  # 输出结果
  print(f"{'Function Name':<55} | {'Count':<6} | {'Avg (μs)':<10} | {'Total (μs)'}")
  print('-' * 90)
  for func, times in times_map.items():
    count = len(times)
    avg_time = sum(times) / count
    total_time = sum(times)
    print(f'{func:<55} | {count:<6} | {avg_time:<10.2f} | {total_time}')


if __name__ == '__main__':
  # 传入文本或文件路径（如 "profile.log"）
  calculate_profile_averages("build/Debug/profile.log")