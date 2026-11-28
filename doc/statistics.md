# Rendering time baseline: No TAAU, cell step size = 1
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




## 总结

| 要回答的问题 | 模拟 | 帧率限制 | 场景 |
|---|---|---|---|
| 哪个步长、哪个 scale 更快（5.0 到 5.4） | **暂停** | **关闭**，或者每次相同 | 2~3 个固定状态 × 固定机位 |
| Volumes 每一步要多少时间 | 运行 | 每次相同 | 任意 |
| 最终用起来流畅吗（5.5） | 运行 | 设为目标值 | 从头完整跑一段 |

# Upscaling Comparison

All cell step size = 1

## 1x

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

## 2x

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

frames 4366, 10.00 s, FPS 436.5, 1% low 325.1, steps 0
paused frames 4366 / 4366 (paused frames have no simulation timings or diagnostics)

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
gpu_volumes              | 0.0061         | 0.096      | 
gpu_ray_march            | 1.0754         | 1.840      | 
gpu_resolve              | 0.1086         | 0.129      | 
gpu_readback             | 0.4212         | 0.732      | 
------------------------------------------------------------------------
cpu_input                | 0.0132         | 0.492      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 1.6480         | 2.647      | 
cpu_upload               | 0.2694         | 0.461      | 
cpu_ui                   | 0.1327         | 0.469      | 
cpu_present              | 0.2223         | 0.659      | 
------------------------------------------------------------------------

## 3x

scene              Buoyant plume
grid               128x192x128
render             768x768
dt                 0.016667
pressure_cycles    2
multigrid_levels   6
gpu                NVIDIA GeForce RTX 3070
diagnostics_interval 1
vram_used_mib      1625
vram_total_mib     8191

frames 5416, 10.00 s, FPS 541.5, 1% low 331.4, steps 0
paused frames 5416 / 5416 (paused frames have no simulation timings or diagnostics)

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
gpu_volumes              | 0.0065         | 0.079      | 
gpu_ray_march            | 0.6640         | 1.408      | 
gpu_resolve              | 0.0755         | 0.092      | 
gpu_readback             | 0.4127         | 0.842      | 
------------------------------------------------------------------------
cpu_input                | 0.0046         | 0.280      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 1.1947         | 2.127      | 
cpu_upload               | 0.2637         | 0.572      | 
cpu_ui                   | 0.1356         | 0.461      | 
cpu_present              | 0.2427         | 1.781      | 
------------------------------------------------------------------------

## 4x

scene              Buoyant plume
grid               128x192x128
render             768x768
dt                 0.016667
pressure_cycles    2
multigrid_levels   6
gpu                NVIDIA GeForce RTX 3070
diagnostics_interval 1
vram_used_mib      1623
vram_total_mib     8191

frames 5650, 10.00 s, FPS 564.9, 1% low 330.1, steps 0
paused frames 5650 / 5650 (paused frames have no simulation timings or diagnostics)

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
gpu_volumes              | 0.0062         | 0.132      | 
gpu_ray_march            | 0.6028         | 1.356      | 
gpu_resolve              | 0.0750         | 0.093      | 
gpu_readback             | 0.4091         | 0.894      | 
------------------------------------------------------------------------
cpu_input                | 0.0031         | 0.191      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 1.1293         | 2.203      | 
cpu_upload               | 0.2624         | 0.579      | 
cpu_ui                   | 0.1348         | 0.528      | 
cpu_present              | 0.2353         | 2.009      | 
------------------------------------------------------------------------

# Upscaling Comparison with Interop

## 1x

scene              Buoyant plume
grid               128x192x128
render             768x768
dt                 0.016667
pressure_cycles    2
multigrid_levels   6
gpu                NVIDIA GeForce RTX 3070
diagnostics_interval 1
vram_used_mib      1644
vram_total_mib     8191

frames 7169, 10.00 s, FPS 716.9, 1% low 301.8, steps 0
paused frames 7169 / 7169 (paused frames have no simulation timings or diagnostics)

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
gpu_volumes              | 0.0034         | 0.035      | 
gpu_ray_march            | 0.6781         | 0.742      | 
gpu_resolve              | 0.1094         | 0.124      | 
gpu_readback             | 0.0164         | 0.030      | 
------------------------------------------------------------------------
cpu_input                | 0.0060         | 0.414      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 0.1791         | 0.507      | 
cpu_upload               | 0.0967         | 0.545      | 
cpu_ui                   | 0.1595         | 0.573      | 
cpu_present              | 0.2333         | 2.772      | 
------------------------------------------------------------------------



## 2x

scene              Buoyant plume
grid               128x192x128
render             768x768
dt                 0.016667
pressure_cycles    2
multigrid_levels   6
gpu                NVIDIA GeForce RTX 3070
diagnostics_interval 1
vram_used_mib      1632
vram_total_mib     8191

frames 9916, 10.00 s, FPS 991.5, 1% low 261.1, steps 0
paused frames 9916 / 9916 (paused frames have no simulation timings or diagnostics)

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
gpu_volumes              | 0.0032         | 0.063      | 
gpu_ray_march            | 0.3377         | 0.456      | 
gpu_resolve              | 0.0766         | 0.089      | 
gpu_readback             | 0.0170         | 0.031      | 
------------------------------------------------------------------------
cpu_input                | 0.0032         | 0.280      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 0.1566         | 0.704      | 
cpu_upload               | 0.0643         | 0.310      | 
cpu_ui                   | 0.1416         | 0.776      | 
cpu_present              | 0.2259         | 3.265      | 
------------------------------------------------------------------------



## 3x

scene              Buoyant plume
grid               128x192x128
render             768x768
dt                 0.016667
pressure_cycles    2
multigrid_levels   6
gpu                NVIDIA GeForce RTX 3070
diagnostics_interval 1
vram_used_mib      1628
vram_total_mib     8191

frames 9261, 8.62 s, FPS 1074.6, 1% low 281.4, steps 0
paused frames 9261 / 9261 (paused frames have no simulation timings or diagnostics)

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
gpu_volumes              | 0.0031         | 0.027      | 
gpu_ray_march            | 0.2737         | 0.388      | 
gpu_resolve              | 0.0759         | 0.091      | 
gpu_readback             | 0.0170         | 0.031      | 
------------------------------------------------------------------------
cpu_input                | 0.0042         | 0.158      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 0.1553         | 0.808      | 
cpu_upload               | 0.0664         | 0.314      | 
cpu_ui                   | 0.1393         | 1.083      | 
cpu_present              | 0.2181         | 3.237      | 
------------------------------------------------------------------------



## 4x

scene              Buoyant plume
grid               128x192x128
render             768x768
dt                 0.016667
pressure_cycles    2
multigrid_levels   6
gpu                NVIDIA GeForce RTX 3070
diagnostics_interval 1
vram_used_mib      1628
vram_total_mib     8191

frames 7957, 7.01 s, FPS 1135.8, 1% low 272.8, steps 0
paused frames 7957 / 7957 (paused frames have no simulation timings or diagnostics)

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
gpu_volumes              | 0.0031         | 0.063      | 
gpu_ray_march            | 0.2261         | 0.281      | 
gpu_resolve              | 0.0758         | 0.090      | 
gpu_readback             | 0.0169         | 0.029      | 
------------------------------------------------------------------------
cpu_input                | 0.0042         | 0.187      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 0.1408         | 0.512      | 
cpu_upload               | 0.0583         | 0.256      | 
cpu_ui                   | 0.1353         | 0.740      | 
cpu_present              | 0.2194         | 3.265      | 
------------------------------------------------------------------------


# Cell Step Size Comparison

All upscaling = 1x

## cell = 2

frames 4951, 10.00 s, FPS 495.1, 1% low 355.2, steps 0
paused frames 4951 / 4951 (paused frames have no simulation timings or diagnostics)

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
gpu_volumes              | 0.0065         | 0.318      | 
gpu_ray_march            | 0.8175         | 1.580      | 
gpu_resolve              | 0.1090         | 0.131      | 
gpu_readback             | 0.4134         | 0.739      | 
------------------------------------------------------------------------
cpu_input                | 0.0060         | 0.346      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 1.3815         | 2.355      | 
cpu_upload               | 0.2679         | 0.604      | 
cpu_ui                   | 0.1359         | 0.521      | 
cpu_present              | 0.2234         | 1.768      | 
------------------------------------------------------------------------

## cell = 4

frames 5306, 10.00 s, FPS 530.5, 1% low 361.0, steps 0
paused frames 5306 / 5306 (paused frames have no simulation timings or diagnostics)

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
gpu_volumes              | 0.0073         | 0.134      | 
gpu_ray_march            | 0.6905         | 1.444      | 
gpu_resolve              | 0.1091         | 0.129      | 
gpu_readback             | 0.4069         | 0.855      | 
------------------------------------------------------------------------
cpu_input                | 0.0059         | 0.459      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 1.2505         | 2.189      | 
cpu_upload               | 0.2662         | 0.460      | 
cpu_ui                   | 0.1336         | 0.495      | 
cpu_present              | 0.2234         | 1.767      | 
------------------------------------------------------------------------

## cell = 8

frames 5406, 10.00 s, FPS 540.6, 1% low 334.2, steps 0
paused frames 5406 / 5406 (paused frames have no simulation timings or diagnostics)

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
gpu_volumes              | 0.0066         | 0.131      | 
gpu_ray_march            | 0.6318         | 1.403      | 
gpu_resolve              | 0.1093         | 0.124      | 
gpu_readback             | 0.4136         | 1.039      | 
------------------------------------------------------------------------
cpu_input                | 0.0062         | 0.483      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 1.1979         | 2.489      | 
cpu_upload               | 0.2653         | 0.607      | 
cpu_ui                   | 0.1371         | 0.532      | 
cpu_present              | 0.2381         | 2.312      | 
------------------------------------------------------------------------

## cell = 16

frames 5495, 10.00 s, FPS 549.4, 1% low 322.7, steps 0
paused frames 5495 / 5495 (paused frames have no simulation timings or diagnostics)

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
gpu_volumes              | 0.0067         | 0.076      | 
gpu_ray_march            | 0.6115         | 1.375      | 
gpu_resolve              | 0.1095         | 0.122      | 
gpu_readback             | 0.4053         | 1.042      | 
------------------------------------------------------------------------
cpu_input                | 0.0019         | 0.123      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 1.1687         | 2.459      | 
cpu_upload               | 0.2625         | 0.615      | 
cpu_ui                   | 0.1357         | 0.491      | 
cpu_present              | 0.2459         | 1.742      | 
------------------------------------------------------------------------

## cell 32

frames 5514, 10.00 s, FPS 551.3, 1% low 326.0, steps 0
paused frames 5514 / 5514 (paused frames have no simulation timings or diagnostics)

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
gpu_volumes              | 0.0067         | 0.093      | 
gpu_ray_march            | 0.5924         | 1.344      | 
gpu_resolve              | 0.1094         | 0.124      | 
gpu_readback             | 0.4089         | 0.968      | 
------------------------------------------------------------------------
cpu_input                | 0.0082         | 0.567      | 
cpu_simulation           | 0.0000         | 0.000      | 
cpu_render               | 1.1538         | 2.182      | 
cpu_upload               | 0.2653         | 0.533      | 
cpu_ui                   | 0.1377         | 0.613      | 
cpu_present              | 0.2434         | 1.877      | 
------------------------------------------------------------------------