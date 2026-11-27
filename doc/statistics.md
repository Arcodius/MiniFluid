# Rendering time baseline
profile_20261126_134602.csv

scene              Buoyant plume
grid               128x192x128
render             768x768
dt                 0.016667
pressure_cycles    2
multigrid_levels   6
gpu                NVIDIA GeForce RTX 3070
diagnostics_interval 1
vram_used_mib      1639
vram_total_mib     8191

frames 3431, 10.00 s, FPS 343.1, 1% low 246.3, steps 0
paused frames 3431 / 3431 (paused frames have no simulation timings or diagnostics)

Stage                    | Avg/frame (ms) | Max (ms)   | Avg/step (ms)
------------------------------------------------------------------------
gpu_advect_velocity      | 0.0000         | 0.000      | 
gpu_sources              | 0.0000         | 0.000      | 
gpu_forces               | 0.0000         | 0.000      | 
gpu_boundary             | 0.0000         | 0.000      | 
gpu_divergence           | 0.0000         | 0.000      | 
gpu_pressure             | 0.0000         | 0.000      | 
gpu_projection           | 0.0000         | 0.000      | 
gpu_advect_scalars       | 0.0000         | 0.000      | 
gpu_diagnostics          | 0.0000         | 0.000      | 
gpu_volumes              | 0.0056         | 0.104      | 
gpu_ray_march            | 1.6653         | 3.033      | 
gpu_resolve              | 0.1088         | 0.123      | 
gpu_readback             | 0.4416         | 0.711      | 
------------------------------------------------------------------------
cpu_input                | 0.0117         | 0.405      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 2.2580         | 3.638      | 
cpu_upload               | 0.2779         | 0.547      | 
cpu_ui                   | 0.1277         | 0.416      | 
cpu_present              | 0.2338         | 0.718      | 
------------------------------------------------------------------------

Charts saved to D:\Projects\MiniFluid\profile_charts


## 总结

| 要回答的问题 | 模拟 | 帧率限制 | 场景 |
|---|---|---|---|
| 哪个步长、哪个 scale 更快（5.0 到 5.4） | **暂停** | **关闭**，或者每次相同 | 2~3 个固定状态 × 固定机位 |
| Volumes 每一步要多少时间 | 运行 | 每次相同 | 任意 |
| 最终用起来流畅吗（5.5） | 运行 | 设为目标值 | 从头完整跑一段 |
