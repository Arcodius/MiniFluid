"""汇总运行时分析器导出的 CSV（F1 窗口中的 Record / Export，或 --profile-record）。

用法: python src/log_profile.py <profile.csv> [输出目录]
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 配置 ----
LOG_PATH = 'profile.csv'
OUTPUT_DIR = 'profile_charts'
WARMUP_RATIO = 0.25  # 丢弃开头的预热比例（首次 kernel 加载会有数百毫秒的尖峰）

# 主曲线：显示名 -> 列名
MAIN_COLUMNS = {
    'frame': 'frame_ms',
    'GPU simulation': 'gpu_sim_ms',
    'GPU rendering': 'gpu_render_ms',
}
# 诊断曲线：显示名 -> (列名, 是否对数坐标)
DIAGNOSTIC_COLUMNS = {
    'max |div| after projection': ('div_after_max', True),
    'relative pressure residual': ('pressure_residual', True),
    'CFL': ('cfl', False),
    'kinetic energy': ('kinetic_energy', False),
}
# 每帧执行一次的渲染阶段；其余 gpu_ 列属于模拟 step，可按步数折算
RENDER_STAGES = ('gpu_volumes_ms', 'gpu_ray_march_ms', 'gpu_readback_ms')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# --------------------------------------------------------------------------


def read_profile(path):
  """返回 (metadata, rows)。rows 中的空单元格（未采样的诊断）为 None。"""
  metadata = {}
  data_lines = []
  for line in Path(path).read_text(encoding='utf-8').splitlines():
    if line.startswith('#'):
      key, _, value = line[1:].partition(':')
      if value:
        metadata[key.strip()] = value.strip()
    elif line.strip():
      data_lines.append(line)
  rows = []
  for raw in csv.DictReader(data_lines):
    rows.append({k: (float(v) if v != '' else None) for k, v in raw.items()})
  return metadata, rows[int(len(rows) * WARMUP_RATIO):]


def column(rows, name):
  return [(r['time_s'], r[name]) for r in rows if r.get(name) is not None]


def stage_columns(rows, prefix):
  names = [k for k in rows[0] if k.startswith(prefix) and k.endswith('_ms')]
  return [n for n in names if n not in ('gpu_sim_ms', 'gpu_render_ms')]


def label_of(name):
  return name.removeprefix('gpu_').removeprefix('cpu_').removesuffix('_ms')


def print_summary(metadata, rows):
  for key, value in metadata.items():
    print(f'{key:<18} {value}')
  frame_ms = [r['frame_ms'] for r in rows]
  steps = sum(r['sim_steps'] for r in rows)
  seconds = sum(frame_ms) / 1000.0
  p99 = sorted(frame_ms)[min(len(frame_ms) - 1, len(frame_ms) * 99 // 100)]
  print(f'\nframes {len(rows)}, {seconds:.2f} s, FPS {len(rows) / seconds:.1f}, '
        f'1% low {1000.0 / p99:.1f}, steps {int(steps)}')

  print(f"\n{'Stage':<24} | {'Avg/frame (ms)':<14} | {'Max (ms)':<10} | Avg/step (ms)")
  print('-' * 72)
  for prefix in ('gpu_', 'cpu_'):
    for name in stage_columns(rows, prefix):
      values = [r[name] for r in rows]
      simulation = prefix == 'gpu_' and name not in RENDER_STAGES
      per_step = f'{sum(values) / steps:.4f}' if simulation and steps else ''
      print(f'{prefix + label_of(name):<24} | {sum(values) / len(values):<14.4f} | '
            f'{max(values):<10.3f} | {per_step}')
    print('-' * 72)

  samples = [r for r in rows if r.get('div_after_max') is not None]
  if samples:
    print(f'\ndiagnostics: {len(samples)} samples (mean / max)')
    for name in ('div_before_max', 'div_after_max', 'div_after_rms', 'pressure_residual',
                 'cfl', 'max_speed', 'kinetic_energy', 'smoke_mass'):
      values = [r[name] for r in samples]
      print(f'  {name:<18} {sum(values) / len(values):<12.4g} {max(values):.4g}')


def plot_curves(series, title, path, ylabel='Time (ms)', log=False):
  fig, ax = plt.subplots(figsize=(12, 5))
  for label, points in series.items():
    if points:
      ax.plot([t for t, _ in points], [v for _, v in points], label=label, linewidth=0.8)
  if log:
    ax.set_yscale('log')
  ax.set_title(title)
  ax.set_xlabel('Time (s)')
  ax.set_ylabel(ylabel)
  ax.grid(alpha=0.3)
  ax.legend(loc='upper right', fontsize=8)
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def plot_pie(rows, path):
  steps = sum(r['sim_steps'] for r in rows)
  names = [n for n in stage_columns(rows, 'gpu_') if n not in RENDER_STAGES]
  items = [(label_of(n), sum(r[n] for r in rows) / steps) for n in names]
  items = sorted([kv for kv in items if kv[1] > 0], key=lambda kv: kv[1], reverse=True)
  total = sum(v for _, v in items)

  fig, ax = plt.subplots(figsize=(10, 7))
  wedges, _, _ = ax.pie(
      [v for _, v in items],
      autopct=lambda p: f'{p:.1f}%' if p >= 2 else '',
      startangle=90,
      counterclock=False,
      pctdistance=0.8,
  )
  ax.legend(
      wedges,
      [f'{l}  {v * 1000:.0f} μs ({v / total:.1%})' for l, v in items],
      loc='center left',
      bbox_to_anchor=(1, 0.5),
      fontsize=9,
  )
  ax.set_title(f'step 内部 GPU 阶段平均耗时占比（{total:.3f} ms/step）')
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def main(log_path=LOG_PATH, out_dir=OUTPUT_DIR):
  metadata, rows = read_profile(log_path)
  if not rows:
    raise SystemExit(f'No profile rows found in {log_path}')
  print_summary(metadata, rows)

  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  plot_curves({k: column(rows, c) for k, c in MAIN_COLUMNS.items()},
              'Frame and GPU time per frame', out / 'main_curve.png')
  stepped = [r for r in rows if r['sim_steps'] > 0]
  if stepped:
    # 只取有模拟步的帧，并按步数折算为每步耗时
    plot_curves({label_of(n): [(r['time_s'], r[n] / r['sim_steps']) for r in stepped]
                 for n in stage_columns(rows, 'gpu_') if n not in RENDER_STAGES},
                'GPU simulation stages per step', out / 'sub_curve.png')
    plot_pie(rows, out / 'sub_pie.png')
  for title, (name, log) in DIAGNOSTIC_COLUMNS.items():
    points = [(t, v) for t, v in column(rows, name) if not log or v > 0]
    if points and not all(math.isnan(v) for _, v in points):
      plot_curves({title: points}, title, out / f'diag_{name}.png', ylabel=title, log=log)
  print(f'\nCharts saved to {out.resolve()}')


if __name__ == '__main__':
  import sys
  main(*sys.argv[1:3])
