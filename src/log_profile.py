from collections import defaultdict
from pathlib import Path
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 配置 ----
LOG_PATH = 'profile.log'
OUTPUT_DIR = 'profile_charts'
WARMUP_RATIO = 0.25  # 丢弃日志开头的预热比例

# 主函数配置表：显示名 -> 日志中函数名需要包含的关键字。
# 命中的函数只出现在“主函数耗时曲线”中；其余全部视为 step 的子函数。
# updateVolumes 每帧只调用一次（与 render 同频，远低于 step），因此也按主函数处理。
MAIN_FUNCS = {
    'step': 'solver.step',
    'render': 'render',
    'updateVolumes': 'updateVolumes',
}
# 子函数的父函数（饼图以它的平均耗时为参照，计算“未统计部分”）
PARENT_FUNC = 'step'

# 日志里的耗时单位 -> μs
UNIT_TO_US = {'μs': 1.0, 'ms': 1000.0}

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# --------------------------------------------------------------------------

# 兼容 [Profile] / [CUDA Profile]，可选的 "Kernel: " 前缀，单位 μs 或 ms
PATTERN = re.compile(
    r'\[(?:CUDA\s+)?Profile\]:?\s*(?:Kernel:\s*)?(.*?)\s*\|\s*Time:\s*([\d.]+)\s*(μs|ms)'
)


def main_name_of(func_name):
  """返回函数对应的主函数显示名，不是主函数则返回 None。"""
  for display, keyword in MAIN_FUNCS.items():
    if keyword in func_name:
      return display
  return None


def short_name(func_name):
  """去掉参数列表，图例更简洁：advectVelocity(dt) -> advectVelocity"""
  return func_name.split('(')[0]


def parse_log(path):
  lines = Path(path).read_text(encoding='utf-8').splitlines()
  lines = lines[int(len(lines) * WARMUP_RATIO):]
  main_times = defaultdict(list)  # 主函数显示名 -> [μs]
  sub_times = defaultdict(list)  # 子函数名 -> [μs]
  for line in lines:
    m = PATTERN.search(line)
    if not m:
      continue
    name = m.group(1).strip()
    value = float(m.group(2)) * UNIT_TO_US[m.group(3)]
    main = main_name_of(name)
    if main:
      main_times[main].append(value)
    else:
      sub_times[short_name(name)].append(value)
  return main_times, sub_times


def print_table(main_times, sub_times):
  print(f"{'Function':<30} | {'Count':<6} | {'Avg (μs)':<12} | Total (μs)")
  print('-' * 70)
  for group in (main_times, sub_times):
    for func, times in group.items():
      print(f'{func:<30} | {len(times):<6} | '
            f'{sum(times) / len(times):<12.2f} | {sum(times):.2f}')
    print('-' * 70)


def plot_curves(times_map, title, path):
  fig, ax = plt.subplots(figsize=(12, 6))
  for func, times in times_map.items():
    # 各函数采样频率不同，但覆盖的总时长相同：横轴归一化为运行进度，使所有曲线等长
    n = len(times)
    x = [i / (n - 1) * 100 if n > 1 else 0 for i in range(n)]
    ax.plot(x, times, label=f'{func} (n={n})', linewidth=1)
  ax.set_xlim(0, 100)
  ax.set_title(title)
  ax.set_xlabel('Run progress after warmup (%)')
  ax.set_ylabel('Time (μs)')
  ax.grid(alpha=0.3)
  ax.legend(loc='upper right', fontsize=8)
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def plot_pie(main_times, sub_times, path):
  avgs = {f: sum(t) / len(t) for f, t in sub_times.items()}
  parent = main_times.get(PARENT_FUNC)
  if parent:
    # 子函数之和之外的开销（同步、拷贝等）单独一块，便于发现未被覆盖的耗时
    other = sum(parent) / len(parent) - sum(avgs.values())
    if other > 0:
      avgs['(other / untracked)'] = other
  items = sorted(avgs.items(), key=lambda kv: kv[1], reverse=True)
  labels = [k for k, _ in items]
  values = [v for _, v in items]
  total = sum(values)

  fig, ax = plt.subplots(figsize=(10, 7))
  wedges, _, autotexts = ax.pie(
      values,
      autopct=lambda p: f'{p:.1f}%' if p >= 2 else '',
      startangle=90,
      counterclock=False,
      pctdistance=0.8,
  )
  ax.legend(
      wedges,
      [f'{l}  {v:.0f} μs ({v / total:.1%})' for l, v in items],
      loc='center left',
      bbox_to_anchor=(1, 0.5),
      fontsize=9,
  )
  ax.set_title(f'{PARENT_FUNC} 内部函数平均耗时占比')
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def main(log_path=LOG_PATH, out_dir=OUTPUT_DIR):
  main_times, sub_times = parse_log(log_path)
  if not main_times and not sub_times:
    raise SystemExit(f'No profile entries found in {log_path}')
  print_table(main_times, sub_times)

  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  if main_times:
    plot_curves(main_times, 'Main functions time', out / 'main_curve.png')
  if sub_times:
    plot_curves(sub_times, 'Sub functions time', out / 'sub_curve.png')
    plot_pie(main_times, sub_times, out / 'sub_pie.png')
  print(f'Charts saved to {out.resolve()}')


if __name__ == '__main__':
  import sys
  main(*sys.argv[1:3])
