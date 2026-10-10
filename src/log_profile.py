from collections import defaultdict
import re

log_data = """
[Profile]: render(cur_ticks) | Time: 2731 μs
[CUDA Profile] Kernel: advectVelocity(dt) | Time: 1.11709 ms
[CUDA Profile] Kernel: addVelocitySource(1.0f) | Time: 0.058208 ms
[Profile]: solver.step(static_cast<float>(simulation_dt)) | Time: 2502 μs
"""


def calculate_mixed_profile(data_source):
  times_map = defaultdict(list)

  # 正则解析：
  # 1. 兼容 [Profile] 和 [CUDA Profile] -> \[(?:CUDA\s+)?Profile\]
  # 2. 兼容可能存在的 "Kernel: " 前缀 -> (?:Kernel:\s*)?
  # 3. 匹配耗时数字和单位 (μs 或 ms)
  pattern = re.compile(
      r'\[(?:CUDA\s+)?Profile\]:?\s*(?:Kernel:\s*)?(.*?)\s*\|\s*Time:\s*([\d.]+)\s*(μs|ms)'
  )

  lines = (
      data_source.strip().splitlines()
      if '\n' in data_source
      else open(data_source, encoding='utf-8').readlines()
  )
  warmup = int(len(lines) * 0.25)
  for line in lines[warmup:]:
    match = pattern.search(line)
    if match:
      func_name = match.group(1).strip()
      time_val = float(match.group(2))
      unit = match.group(3)

      # 如果是 ms，统一转换为 μs (乘以 1000) 方便对比
      if unit == 'ms':
        time_val *= 1000

      times_map[func_name].append(time_val)

  # 输出结果
  print(
      f"{'Function / Kernel Name':<50} | {'Count':<6} | {'Avg (μs)':<12} | '{'Total (μs)'}'"
  )
  print('-' * 95)
  for func, times in times_map.items():
    count = len(times)
    avg_time = sum(times) / count
    total_time = sum(times)
    print(f'{func:<50} | {count:<6} | {avg_time:<12.2f} | {total_time:.2f}')


if __name__ == '__main__':
  calculate_mixed_profile("build/Debug/profile.log")