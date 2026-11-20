# Render: What to Do Next

我通读了 [cuda_renderer.cu](src/cuda/cuda_renderer.cu)，下面是按预期收益从大到小排的加速建议，后面是 TAA 的做法。这些都是读代码得出的判断，我没有实际跑 profiler。建议先跑一次，看 `GpuStage::Volumes`、`RayMarch`、`Readback` 三段各占多少时间，再决定先改哪个。

## 一、加速建议

### 1. 光照体积（`lightDepthKernel`）：大部分结果可以缓存
每帧每个体素要做三次 march：
- `solidVisibility(sun)`：最多 256 步，只跟固体和光照方向有关。
- `solidVisibility(up)`：只跟固体有关，可以说是常量。
- `opticalDepthToLight`：最多约 1.7N 步，这是唯一真正需要每帧算的。

而且 [cuda_renderer.cu:670](src/cuda/cuda_renderer.cu#L670) 每帧都在执行 `sdf_.copyFrom`。

**改法**：
- 把 `.z/.w`（固体可见度）拆到一张单独的 `float2` 体积里，加一个 dirty 标记，只在固体或光照角度变化时重算。SDF 拷贝也走同样的 dirty 逻辑。
- 烟雾的光学深度可以改成逐切片扫描，类似现在的 `skyDepthKernel`：沿光线的主轴逐层推进，每层从上一层对应位置做双线性插值，再加上本段的贡献。这样复杂度从 O(N⁴) 降到 O(N³)，不再是每个体素各自 march 整条光线。
- 更简单的退路是把光照体积降到半分辨率（烟雾的光照本身很平滑），或者每两帧更新一次。

### 2. 主 ray march：跳过空白区域
步长是 0.5 个格子，大部分空间却是空的。可以每帧用一个 kernel 生成粗网格，比如每 8³ 一块，存这块的 max density（用 `fmaxf` 做规约）。march 时先查粗网格，遇到空块就用 DDA 直接跳到块的出口。烟雾越稀疏收益越大，通常有 2–5 倍。

### 3. 地面和固体着色里的逐像素 march
- `floorColor` 里每个像素又跑一次 `opticalDepthToLight`（最多 1024 步）和一次 `solidVisibility`（256 步）。
- `solidColor` 里也有一次 `opticalDepthToLight`。

**改法**：
- 固体表面的烟雾阴影直接采样 `lightDepth.x`，在 `p + normal*cellSize` 处取，因为这些值已经算好了。
- 地面可以在光照体积更新时顺带生成一张 2D 阴影图：覆盖体积沿光线方向投影到地面的范围，存 `exp(-tau)*vis`。像素着色时只需要一次 `tex2D`。

### 4. 小的算术和带宽优化
- 三个 octave 的 `expf(-x)`、`expf(-x/2)`、`expf(-x/4)` 可以改成一次 `expf` 加两次 `sqrtf`。还可以加 `--use_fast_math`，或者手动用 `__expf`。
- density 和 temperature 合成一张 `float2` 纹理，开启 emission 时每步少一次 fetch。
- 体积纹理改用 half（`cudaChannelFormatKindFloat` 16bit），`lightDepth` 也改成 half4，带宽减半，精度对烟雾来说足够。

### 5. 读回和上传
现在的流程是 `cudaMemcpy`（同步，可分页内存）→ CPU → `SDL_UpdateTexture` → GPU。768² 的图来回两趟，而且 CPU 要等 GPU。
- 最小改动：`gFrameBuffer` 改用 `cudaMallocHost` 分配的锁页内存。
- 根本的解决办法：拿到 SDL 后端的 D3D11 或 GL 纹理，用 `cudaGraphicsD3D11RegisterResource` 或 GL 的 interop，让 CUDA 直接写显示纹理，完全不经过 CPU。

### 6. 降分辨率渲染
这一条和 TAA 一起做最划算：半分辨率或棋盘格渲染，再用 TAA 做上采样（TAAU），ray march 开销直接降到 1/4–1/2。

## 二、TAA 怎么做

体积渲染做 TAA 有两个特殊点：
- 一个像素没有唯一的深度。
- 烟雾自己在动，相机不动时也一样。

### 1. 把噪声变成随帧变化
现在 `gradientNoise(x, y)` 是固定的空间图案。改成随帧号变化：

```cpp
// 每帧偏移，IGN 的常用做法
const float jitter = gradientNoise(x + 5.588238f * (frame & 63), y + 5.588238f * (frame & 63));
// 子像素抖动（给固体边缘抗锯齿用）：Halton(2,3)
ndcX = 2.0f * (x + 0.5f + jx) / width - 1.0f;  // jx, jy ∈ [-0.5, 0.5]
```

这样做之后步长可以从 0.5 格放宽到 1 格，噪声交给 TAA 去积分，本身就是一次加速。

### 2. 把渲染拆成三个 pass
1. **`renderKernel`**：输出线性 HDR 的 `float4`，rgb 是颜色，a 是代表深度。不在这里做 tonemap 和 dither。
2. **`taaResolveKernel`**：读当前帧、上一帧的 history 和深度，写出新的 history。两张 history buffer 交替使用。
3. **`tonemapKernel`**：history 经过 tonemap 和 dither，输出 `uint8`。

### 3. 体积的代表深度
march 时按每一步的贡献权重求平均深度：

```cpp
float w = transmittance * (1.0f - stepTransmittance);
depthSum += w * distance;  weightSum += w;
// 结束后：
float surface = fminf(solidDistance, floorDistance);       // 天空记为 FLT_MAX
float depth = weightSum > 0.3f ? depthSum / weightSum : surface;
```

### 4. 重投影
CPU 端保存上一帧的 `RenderCamera`，作为 `prevCamera` 传进来：

```cpp
float3 P = origin + direction * depth;            // 天空像素只用方向：P = prevPos + direction
float3 d = P - prev.position;
float z  = dot(d, prev.front);
float px = dot(d, prev.right) / (z * prev.tangentHalfFov * aspect);
float py = dot(d, prev.up)    / (z * prev.tangentHalfFov);
// 正交相机：px = dot(d, right) / halfWidth，py 同理
float2 prevPix = make_float2((px + 1) * 0.5f * w - 0.5f, (1 - py) * 0.5f * h - 0.5f);
```

可选：减去烟雾自身的运动，即 `P -= velocity(P) * dt`，速度场直接从 solver 采样。这对快速翻滚的烟雾很有帮助。

### 5. 解析（resolve）
- history 用 bilinear 采样，或者用 5-tap Catmull-Rom，后者更锐利。
- 在当前帧 3×3 邻域里用 YCoCg 空间算均值 μ 和标准差 σ，把 history 裁剪（clip）到 `μ ± γσ` 范围内，γ 取 1.0–1.25。这一步最关键，它负责抑制烟雾运动和遮挡变化带来的鬼影。
- 混合：`out = lerp(history, current, α)`，α 约 0.1。重投影落到屏幕外时 α=1；深度差异大（遮挡变化）时提高 α；暂停模拟时可以把 α 降到 0.05，让画面收敛得更干净。
- 混合前对颜色做 `c / (1 + luma(c))` 加权，混合后再还原（Karis 的做法），可以防止高亮的太阳光晕闪烁。
- 分辨率改变、光照参数改变时清空 history。

### 6. 进阶
- **TAAU**：按 0.5×–0.7× 分辨率渲染，resolve 时输出到全分辨率。
- **光照体积的时间摊销**：光线 march 的起点也随帧号抖动，光照体积在 3D 里做指数混合（α≈0.3）。这样可以用更少的步数。

如果需要，我可以先实现 TAA 的最小版本：随帧抖动、HDR 中间 buffer、相机重投影加方差裁剪，再在这个基础上把步长放宽。也可以先做第 1 条的光照缓存，它的改动最小、收益最确定。