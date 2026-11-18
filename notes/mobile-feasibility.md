# 鸟类识别模型上手机：可行性调研

## 结论（5 行）

1. **iOS：路径齐全，可行但未实测。** `coremltools` 能把 ViT 转成 ML Program，支持 8-bit per-channel 权重（iOS 16+），和我们 0.63 GB 文件是同一种方案（要用 `coremltools` 重新量化并重测 top-1）。估算延迟每张 0.1–1 秒（不是测量；没找到 ViT-H 在 iPhone 上的一手数字）。
2. **Android：只对旗舰机现实。** 找到的一手数据只有 ViT-B/16 在 Snapdragon NPU 上约 3–6 ms，没找到 ViT-L / ViT-H 的官方数据。估算旗舰每张 0.05–0.6 秒，CPU 保底数秒（不是测量）。
3. **推荐做法：** 先做 iOS（Core ML，8-bit 或 6-bit palettization，双输出，模型用 Background Assets 下发）；Android 用 ExecuTorch（XNNPACK 保底，QNN 加速）或 ONNX Runtime + QNN。同时保留「Mac 上算，手机只看结果」作为不依赖真机测量的保底方案。
4. **最大风险：内存，不是速度。** 权重在内存里可能按 float16 展开成 1.26 GB，再加上检测器 OWLv2 的 0.62 GB，容易被系统杀掉。其次是 ANE / NPU 不接这么大的模型，退回 GPU 或 CPU。
5. **真机上必须先量的：** 每张图延迟（分 compute unit）、峰值内存、首次加载和编译时间、8-bit / float16 在真机上的 top-1 是否还是 92.4%、连续跑几百张后的发热降速。

调研日期：2026-11-17。只读了网页，没有下载任何权重，没有在真机上跑过。
文中凡是标了「估算」的数字都不是测量值。标了「未核实」的说法没有找到一手来源。

## 一、iOS：Core ML

### 步骤

1. **转换（convert）**
   - 两步：先用 `torch.jit.trace` 或 `torch.export.export` 抓图，再用 `ct.convert` 转。官方说 `torch.jit.trace` 是「稳定且性能更好」的路径，目前推荐用它；`torch.export` 路径从 Core ML Tools 8.0 起加入，还是 beta，官方列出的已验证模型里有 ViT。
     来源：https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html
   - 默认输出格式是 ML Program（`.mlpackage`），需要 iOS 15+。权重和程序结构分开存放。
     来源：https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html
   - 精度：ML Program 默认 float16。GPU 和 Neural Engine 用 float16 计算；只有 CPU-only 能保证 float32。
     来源：https://apple.github.io/coremltools/docs-guides/source/typed-execution.html
   - 我们在 Mac 上已经量过 float16 精度不掉，这一条对我们有利。但 Mac GPU 的 float16 和 iPhone ANE 的 float16 不是同一套实现，真机上要重测。

2. **压缩（compress）**。各选项和需要的系统版本（来源：https://apple.github.io/coremltools/docs-guides/source/opt-whats-new.html ）：

   | 选项 | 最低系统 | 0.63 B 参数的大致文件大小（估算，参数数 x 位数） |
   |---|---|---|
   | float16（默认） | iOS 15 | 1.26 GB（和我们现有文件同大小） |
   | 线性 8-bit，weight-only，per-channel scale | iOS 16 | 0.63 GB（和我们的 per-row int8 是同一种方案；Core ML 版要从浮点模型重新量化，top-1 要重测） |
   | Palettization 1/2/4/6/8-bit，per-tensor LUT | iOS 16 | 6-bit 约 0.47 GB；4-bit 约 0.32 GB |
   | 激活量化 W8A8（A17 Pro / M4 的 NE 上有加速） | iOS 17 | 0.63 GB，需要校准数据 |
   | Palettization per-grouped-channel、3-bit、int8 LUT、per-channel scale | iOS 18 | 同上各位数 |
   | 线性 4-bit 权重、per-block scale | iOS 18 | 约 0.32 GB |

   - API：`coremltools.optimize.coreml.linear_quantize_weights`、`palettize_weights`，都可以不用数据直接对转好的模型做。
     来源：https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html 、https://apple.github.io/coremltools/docs-guides/source/opt-palettization-algos.html
   - 官方说法：8-bit per-channel 的 weight-only 量化「通常能很好地保住精度」；压到 4-bit 需要 per-block 才能保住精度。
     来源：https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html
   - 官方建议：palettization（1 到 8 bit）在 Neural Engine 上的内存和延迟收益通常最好；4-bit per-block 量化在 GPU 上内存收益大；跑在 NE 上建议用 per-channel scale。
     来源：https://apple.github.io/coremltools/docs-guides/source/opt-overview.html 、https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html
   - **文件小不等于内存小。** iOS 16 的压缩权重是「加载前解压成 float16」；iOS 17 起「某些情况下」边算边解压。官方原话是解压策略因硬件和 compute unit 而异，强烈建议在具体模型和芯片上测。
     来源：https://apple.github.io/coremltools/docs-guides/source/opt-whats-new.html 、https://apple.github.io/coremltools/docs-guides/source/opt-overview.html

3. **打包和下发（package / deliver）**
   - iOS app 解压后最大 4 GB，所以 0.63 GB 的模型直接放进 app 包在规则上是允许的。
     来源：https://developer.apple.com/help/app-store-connect/reference/app-uploads/maximum-build-file-sizes
   - On-Demand Resources：单个 tag 最大 512 MB，0.63 GB 的 8-bit 模型放不进一个 tag（6-bit 约 0.47 GB 可以，估算）。而且 Apple 文档已把 On-Demand Resources 称为 deprecated，建议迁到 Background Assets。
     来源：https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/On_Demand_Resources_Guide/PlatformSizesforOn-DemandResources.html 、https://developer.apple.com/documentation/backgroundassets
   - Background Assets：框架本身 iOS 16+。Managed / Apple-hosted（Apple 代为托管，系统管下载、更新、压缩）要求 app 面向 iOS 26+。Apple-hosted 的总量上限 200 GB、最多 200 个 asset pack。可以设成首次启动前必须下完（essential）、预取、或按需。
     来源：https://developer.apple.com/help/app-store-connect/manage-asset-packs/overview-of-apple-hosted-asset-packs/ 、https://developer.apple.com/help/app-store-connect/reference/app-uploads/apple-hosted-asset-pack-size-limits
   - 建议：模型作为一个按需或预取的 asset pack，app 本体保持很小。

4. **运行（run）**
   - 用 `MLComputeUnits` 选 CPU / GPU / Neural Engine。`all` 让系统自己挑。
     来源：https://developer.apple.com/documentation/coreml/mlcomputeunits
   - `MLComputePlan`（iOS 17.4+）能在跑之前看到模型的计算计划，用来确认每一层实际落在哪。Xcode 的 Performance 页签能在连着的真机上出延迟报告。
     来源：https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n 、https://machinelearning.apple.com/research/neural-engine-transformers
   - **632 M 参数能不能上 ANE：没找到 Apple 的答案。** 找到的相关一手说法只有这些：
     - Apple 的 ANE transformer 指南要求改模型结构：`nn.Linear` 换成 1x1 `nn.Conv2d`，张量排成 `(B, C, 1, S)`，按 attention head 拆分。它的案例是 distilbert，不是大 ViT。ANE 最多支持 5 维张量，最后一维按 64 字节对齐。
       来源：https://machinelearning.apple.com/research/neural-engine-transformers 、https://machinelearning.apple.com/research/vision-transformers 、https://github.com/apple/ml-ane-transformers
     - Apple 的 Stable Diffusion 仓库写明：在 iOS / iPadOS 上用 Neural Engine 时，如果权重没有压到 6-bit 或更低，UNet 必须切成两块，每块权重小于 1 GB；macOS 不需要。这是该仓库针对 UNet 的说明，是否是 ANE 的通用上限，文档没说。
       来源：https://github.com/apple/ml-stable-diffusion
     - 按这条类推（推断，不是 Apple 的结论）：我们的 float16 版本 1.26 GB 超过 1 GB，要上 ANE 就得切成两段（比如前 16 层和后 16 层，正好在 block 15 附近切，还顺带给出中间输出），或者压到 6-bit 以下。8-bit 的 0.63 GB 是否算「小于 1 GB」取决于系统是否在加载时把它展开成 float16，未核实。
     - Apple 在 Mac 上跑 Llama-3.1-8B 时明确选了 GPU，理由是这类大模型受内存带宽限制，GPU 的算力和带宽组合最好。那是 Mac 不是 iPhone，但说明大 transformer 走 GPU 是 Apple 自己认可的路线。
       来源：https://machinelearning.apple.com/research/core-ml-on-device-llama
   - 实际做法：同一个 `.mlpackage` 分别用 `cpuAndNeuralEngine` 和 `cpuAndGPU` 各量一遍，哪个快用哪个。不要先花时间按 ANE 指南重写模型，先量原样转换的结果。

5. **暴露 block 15 的 token（第二个输出）**
   - open_clip 的 `VisionTransformer` 有现成的 `forward_intermediates(x, indices=..., output_fmt='NLC', ...)`，能取指定 block 的输出。
     来源：https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
   - `ct.convert` 的 `outputs` 参数对 PyTorch 模型的说明是：列表长度必须等于模型返回的输出个数，可以给每个输出命名。所以包一层 wrapper，返回 `(embedding, block15_tokens)` 两个张量即可。
     来源：https://apple.github.io/coremltools/source/coremltools.converters.convert.html
   - 三个小头（1.3 MB 加几 KB）和 11,172 x 1024 的名字矩阵（float16 约 23 MB）可以放进同一个模型，也可以在 app 里用 Accelerate 算，都很小。
   - 这一步没有实测。多一个中间输出可能挡住转换器的某些图融合优化，未核实。

### 检测器

OWLv2 base（0.62 GB）也得在手机上跑，或者换小检测器。Apple 的 Core ML 模型库页面里没找到 OWLv2 或 OWL-ViT；有 YOLOv3、DETR 这类现成模型。手机上先用小检测器、只把框出的鸟送进大模型，是更现实的组合。来源：https://developer.apple.com/machine-learning/models/

## 二、Android

Android 没有一个统一答案，取决于芯片。三套运行时对比：

| 运行时 | PyTorch 转换路径 | 后端 | 量化 | 有没有大 ViT 的官方路径 |
|---|---|---|---|---|
| ExecuTorch | `torch.export` 后 `to_edge_transform_and_lower`，产出 `.pte` | XNNPACK（CPU）、Vulkan（GPU）、Qualcomm（NPU）、MediaTek（NPU） | PT2E（torchao）。XNNPACK：8-bit，fp32 / fp16 激活。Vulkan：linear 层 8-bit / 4-bit 权重，激活可保持 fp32 / fp16。QNN：8a8w（默认）、16a16w、16a8w、16a4w、16a4w_block，或 fp16。MediaTek：A16W16、A16W8、A16W4、A8W8、A8W4 | 官方例子有 `torchvision_vit`（是 `vit_b_16`）和 `dinov2`。没有 ViT-H。但同一仓库的 Llama 例子在手机上跑过 2.4 GB 和 6.1 GB 的 `.pte`，说明 GB 级文件本身不是障碍 |
| LiteRT（原 TensorFlow Lite） | `litert-torch`（原 ai-edge-torch），基于 `torch.export`，产出 `.tflite`。转换器是 Beta | CPU、GPU delegate、NPU（Qualcomm、MediaTek、Google Tensor） | PT2E（`PT2EQuantizer`）。GPU delegate 只做浮点（16 / 32 bit），8-bit 模型在 GPU 上是把权重解量化回浮点再算 | 官方快速上手是 ResNet18。没找到 ViT-H 的官方例子。NPU 文档说大模型建议 AOT 编译，并可配合 Play AI Pack 下发 |
| ONNX Runtime Mobile | 先导出成 `.onnx`（导出步骤本次没查文档，未核实） | XNNPACK（CPU）、QNN（HTP 即 NPU，或 GPU）、NNAPI | 静态 / 动态 8-bit；`MatMul` 的 block-wise int4（weight-only）；float16 转换。QNN HTP 默认可把 float32 模型按 fp16 跑 | 高通 AI Hub 的 ViT-B/16 和 OWL-ViT 都提供 ONNX + QNN 版本并有实测延迟，是三者里离我们最近的官方先例 |

来源：
- ExecuTorch 后端总览：https://docs.pytorch.org/executorch/stable/backends-overview.html
- XNNPACK：https://docs.pytorch.org/executorch/stable/backends/xnnpack/xnnpack-overview.html
- Vulkan：https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-overview.html
- Qualcomm：https://docs.pytorch.org/executorch/stable/backends-qualcomm.html
- MediaTek（要求 Dimensity 9300 / 9400）：https://docs.pytorch.org/executorch/stable/backends-mediatek.html
- 量化总览：https://docs.pytorch.org/executorch/stable/quantization-overview.html
- 例子目录：https://github.com/pytorch/executorch/tree/main/examples/models ，ViT 例子：https://github.com/pytorch/executorch/blob/main/examples/models/torchvision_vit/model.py ，Llama：https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md
- litert-torch：https://github.com/google-ai-edge/litert-torch ，转换器文档：https://github.com/google-ai-edge/litert-torch/blob/main/docs/pytorch_converter/README.md
- LiteRT GPU：https://ai.google.dev/edge/litert/performance/gpu ，NPU：https://ai.google.dev/edge/litert/next/npu
- ONNX Runtime：NNAPI https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html ，QNN https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html ，XNNPACK https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html ，量化 https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html ，float16 https://onnxruntime.ai/docs/performance/model-optimizations/float16.html
- 高通 AI Hub：https://huggingface.co/qualcomm/VIT 、https://huggingface.co/qualcomm/OWL-ViT

**NNAPI 的状态：** Android 官方写明 NNAPI 在 Android 8.1 引入，在 Android 15 弃用，迁移方向是 Google Play Services 里的 TensorFlow Lite 加 GPU delegate。所以不要基于 ONNX Runtime 的 NNAPI execution provider 做新东西。
来源：https://developer.android.com/ndk/guides/neuralnetworks/migration-guide

### 步骤（推荐 ExecuTorch，理由：和我们的 PyTorch 代码最近，一个 `.pte` 可以带 CPU 保底）

1. **转换**：`torch.export.export(wrapper, sample)`，再 `to_edge_transform_and_lower(..., partitioner=[...])`，`.to_executorch()` 写出 `.pte`。后端不支持的算子会退回 CPU；可以按优先级列多个 partitioner。
   来源：https://docs.pytorch.org/executorch/stable/backends-overview.html 、https://docs.pytorch.org/executorch/stable/backends/xnnpack/xnnpack-overview.html
2. **压缩**：
   - 最稳：fp16（QNN 的 `use_fp16`，XNNPACK / Vulkan 的 fp16 模式），文件 1.26 GB。
   - weight-only 8-bit（对应我们的 0.63 GB 文件）：Vulkan 文档明确支持「8-bit / 4-bit 权重加浮点激活」的 linear 层。XNNPACK 只写了「支持 8-bit 量化」，是否有 weight-only 加浮点激活这一种，未核实。
   - 全整型（8a8w / 16a8w）是 NPU 上最快的形态，但需要校准数据，ViT 做全整型量化有掉点风险（掉点风险这句未核实，需要自己用 eval 集量）。
   来源：https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-overview.html 、https://docs.pytorch.org/executorch/stable/backends-qualcomm.html
3. **打包和下发**：Play 的 base module 压缩后上限 500 MB，0.63 GB 的模型放不进去。用 asset pack 或 AI pack（单个上限 1.5 GB），可选 install-time / fast-follow / on-demand。Play for On-device AI 目前是 beta，可以按设备型号或 RAM（例如「至少 6 GB」）定向下发不同的包，这正好用来「旗舰发大模型，其余发蒸馏小模型」。
   来源：https://support.google.com/googleplay/android-developer/answer/9859372 、https://developer.android.com/google/play/on-device-ai
4. **运行**：先用 XNNPACK（CPU）跑通并对齐精度，再逐个试 QNN（Snapdragon）、Vulkan（GPU）、MediaTek。每个后端单独量延迟和内存。
   来源：https://docs.pytorch.org/executorch/stable/backends-overview.html 、https://docs.pytorch.org/executorch/stable/backends-qualcomm.html 、https://docs.pytorch.org/executorch/stable/backends-mediatek.html
5. **暴露 block 15 的 token**：和 iOS 一样，wrapper 返回两个张量；`torch.export` 支持多输出。ExecuTorch 文档里没找到专门讲多输出的页面，未实测。litert-torch 文档里有「Multi-Signature Conversion」一节可做同样的事。
   来源：https://github.com/google-ai-edge/litert-torch/blob/main/docs/pytorch_converter/README.md

### 检测器

高通 AI Hub 有 OWL-ViT（不是 OWLv2；153 M 参数，768x768，float 613 MB）的实测：Snapdragon 8 Elite NPU 上 17–26 ms，8 Elite Gen 5 上 12–21 ms（因运行时而异）。这说明和我们检测器同量级的模型能整个跑在 NPU 上。OWLv2 base 本身没有找到官方数据。
来源：https://huggingface.co/qualcomm/OWL-ViT

## 三、硬限制

| 项目 | 限制 | 来源 |
|---|---|---|
| iOS app 内存上限 | 没找到 Apple 公布的具体数值。系统在内存紧张时杀 app（jetsam）。上限「在 app 生命周期内会变，不一定对应物理内存」 | https://developer.apple.com/documentation/xcode/identifying-high-memory-use-with-jetsam-event-reports 、https://developer.apple.com/documentation/os/os_proc_available_memory |
| iOS 提高内存上限 | `com.apple.developer.kernel.increased-memory-limit` entitlement，iOS 15+。只在「部分机型」有效，没拿到额外内存时 app 也必须能正常工作。用 `os_proc_available_memory()` 查剩余可用量 | https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit |
| iOS app 包大小 | 解压后最大 4 GB；可执行文件 `__TEXT` 段合计 500 MB | https://developer.apple.com/help/app-store-connect/reference/app-uploads/maximum-build-file-sizes |
| iOS On-Demand Resources | 单 tag 512 MB；同时在用 2 GB；托管总量 20 GB。已被标为 deprecated | https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/On_Demand_Resources_Guide/PlatformSizesforOn-DemandResources.html 、https://developer.apple.com/documentation/backgroundassets |
| iOS Background Assets（Apple 托管） | 总量 200 GB，最多 200 个包。Managed / Apple-hosted 需要 iOS 26+；框架本身 iOS 16+ | https://developer.apple.com/help/app-store-connect/reference/app-uploads/apple-hosted-asset-pack-size-limits 、https://developer.apple.com/help/app-store-connect/manage-asset-packs/overview-of-apple-hosted-asset-packs/ |
| iOS 蜂窝网络下载上限 | 未核实。没有找到 Apple 的一手文档 | 见文末 |
| Google Play base module | 压缩下载 500 MB | https://support.google.com/googleplay/android-developer/answer/9859372 |
| Google Play 单个 feature module | 500 MB | 同上 |
| Google Play 单个 asset pack / AI pack | 1.5 GB | 同上；https://developer.android.com/google/play/on-device-ai |
| Google Play 所有 module 加 install-time 包合计 | 4 GB | 同上 |
| Google Play on-demand / fast-follow 包合计 | 30 GB | 同上 |
| Google Play 移动数据提示 | app 超过 200 MB 时，用移动数据安装会弹一个不阻断的提示 | 同上 |
| Android 每 app 内存 | Java heap 有硬上限，随设备 RAM 而变。模型权重在 native 内存里，native 内存的上限文档没给数值 | https://developer.android.com/topic/performance/memory-overview |
| 当前手机 RAM：Pixel | Pixel 11：12 GB；11 Pro / Pro XL：12 或 16 GB。Pixel 10：12 GB，10 Pro：16 GB。Pixel 10a / 9a：8 GB | https://support.google.com/pixelphone/answer/7158570 |
| 当前手机 RAM：Samsung | Galaxy S26 Ultra：12 或 16 GB | https://www.samsung.com/us/smartphones/galaxy-s26-ultra/ |
| 当前手机 RAM：iPhone | 未核实。Apple 的规格页不写 RAM | https://www.apple.com/iphone-17-pro/specs/ |

内存参考点（一手，但都不是我们的模型）：
- Apple Stable Diffusion：生成过程峰值内存「可超过 2 GB」；iOS 上建议开 `reduceMemory`；app 崩溃时建议加 Increased Memory Limit。来源：https://github.com/apple/ml-stable-diffusion
- ExecuTorch Llama 3.2 1B（BF16，`.pte` 2,358 MiB）在 OnePlus 12 上 RSS 3,185 MiB。也就是常驻内存约等于文件大小再加 0.8 GB 左右。来源：https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md

按这个比例粗估我们的内存（估算）：识别模型 0.63–1.26 GB 加检测器 0.62 GB 加运行时开销，合计大约 1.5–2.5 GB。在 12–16 GB 的 Android 旗舰上问题不大；在 8 GB 机型和 iPhone 上必须真机量。

## 四、延迟参考点和估算

### 先核对计算量

timm 官方 benchmark 表（单位是 GMACs，乘加次数；常被口头叫做 GFLOPs）：

| 模型 | 输入 | 参数 (M) | GMACs | RTX 4090 吞吐 (img/s, AMP) |
|---|---|---|---|---|
| `vit_small_patch16_224` | 224 | 22.05 | 4.61 | 7693 |
| `vit_base_patch16_224` | 224 | 86.57 | 17.58 | 2787 |
| `vit_large_patch14_clip_224` | 224 | 304.2 | 81.08 | 658 |
| `vit_huge_patch14_clip_224` | 224 | 632.05 | 167.4 | 346 |

来源：https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-infer-amp-nchw-pt240-cu124-rtx4090.csv

任务里给的 167 / 17.6 / 4.6 和表一致。ViT-H/14 的计算量是 ViT-B/16 的 9.5 倍，是 ViT-L/14 的 2.06 倍。在 4090 上实测吞吐比是 8.05 倍和 1.90 倍，和计算量比例基本吻合。

### 手机上的一手参考点

| 模型 | 规模 | 设备 | 延迟 | 条件 | 来源 |
|---|---|---|---|---|---|
| MobileCLIP-B 图像塔（ViT-B/16，224） | 86.3 M | iPhone 12 Pro Max | 10.4 ms | Core ML Tools 7.0，iOS 17.0.3，batch 1（v1 论文的测量条件） | https://github.com/apple/ml-mobileclip 、https://arxiv.org/abs/2311.17049 |
| ViT-L/14 图像塔，224（MobileCLIP2-L/14 和 DFN-L/14 两行相同） | 304.3 M | iPhone 12 Pro Max | 57.9 ms | MobileCLIP2 论文表 8；论文只写「在 iPhone12 Pro Max 上测」，工具和系统版本没写 | https://arxiv.org/html/2508.20691v1 、https://github.com/apple/ml-mobileclip |
| SigLIP-L/16，256 | 316.0 M | iPhone 12 Pro Max | 38.2 ms | 同上 | https://arxiv.org/html/2508.20691v1 |
| MobileCLIP2-S4 图像塔，256 | 321.6 M | iPhone 12 Pro Max | 19.6 ms | 同上 | 同上 |
| FastViT MA36 | 42.7 M | iPhone 13 Pro Max / 16 Pro | 4.47 / 2.78 ms | float16，compute unit all | https://developer.apple.com/machine-learning/models/ |
| Depth Anything V2 Small | 24.8 M | iPhone 15 Pro Max / 16 Pro | 33.90 / 26.21 ms | float16，compute unit all | 同上 |
| distilbert（ANE 优化版），序列长 128 | 未列 | iPhone 13 | 3.47 ms | ANE | https://machinelearning.apple.com/research/neural-engine-transformers |
| Stable Diffusion 2.1 base UNet | 未列 | iPhone 14 Pro Max | 2.69 iter/s | CPU_AND_NE，iOS 17 beta | https://github.com/apple/ml-stable-diffusion |
| SDXL UNet（4.04-bit 混合 palettization） | 未列 | iPhone 15 Pro Max | 0.85 iter/s | CPU_AND_NE，768x768 | 同上 |
| ViT（ViT-B/16，224） | 86.6 M | Snapdragon 8 Gen 3 / 8 Elite / 8 Elite Gen 5 | 6.33 / 4.42 / 3.43 ms | ONNX float，NPU | https://huggingface.co/qualcomm/VIT |
| ViT（同上，w8a16） | 86.6 M | Snapdragon 8 Elite / 8 Elite Gen 5 | 3.16 / 2.31 ms | ONNX，NPU | 同上 |
| OpenAI CLIP ViT-B/16（150 M 是两塔参数之和；延迟是否含文本塔，页面摘要没说清） | 150 M | Snapdragon 8 Elite | 14.4 ms | ONNX float，NPU | https://huggingface.co/qualcomm/OpenAI-Clip |
| OWL-ViT，768x768 | 153 M | Snapdragon 8 Elite | 17.1–26.0 ms | float，NPU | https://huggingface.co/qualcomm/OWL-ViT |
| LiteRT 的 72 个模型总体 | — | Snapdragon 8 Elite Gen 5 | NPU 比 CPU 最高快 100 倍，比 GPU 最高快 10 倍 | Google 自测 | https://developers.googleblog.com/en/unlocking-peak-performance-on-qualcomm-npu-with-litert/ |
| 我们自己：ViT-H/14 | 632 M | Mac（Apple silicon，MPS） | 33–40 img/s，即每张 25–30 ms | 批量吞吐，不是手机 | 本项目已测 |

没有找到 ViT-H 在手机上的一手延迟数字（查过的地方见文末）。

### 估算（不是测量）

**iPhone。** 最近的锚点是同一台 iPhone 12 Pro Max 上的 ViT-B/16（10.4 ms）和 ViT-L/14（57.9 ms）。
- 这两点之间，计算量涨 4.61 倍，延迟涨 5.57 倍。延迟涨得比计算量快约 1.2 倍。
- ViT-H/14 的计算量是 ViT-L/14 的 2.06 倍。按纯计算量外推：57.9 x 2.06 ≈ 120 ms。再乘同样的 1.2：约 145 ms。
- iPhone 12 Pro Max 是几代以前的机型。Apple 模型库里同一个模型从 iPhone 13 Pro Max 到 16 Pro 快了约 1.6 倍（4.47 到 2.78 ms）。新机上乐观值大约 **70–150 ms 每张**。
- 这个外推的前提是 ViT-H 能和 ViT-L 走同一条执行路径。如果因为超过 1 GB 而被切块、或者退回 GPU / CPU，可能慢几倍。悲观值按 **0.3–1 秒每张**。
- 合起来：**iPhone 上每张 0.1–1 秒量级**，比 Mac 上的 25–30 ms 慢 3 到 30 倍。两个锚点可能不是同一次、同一套工具版本测的，这也是误差来源。

**Android 旗舰（Snapdragon 8 Elite 一代）。** 锚点是高通的 ViT-B/16：4.42 ms（8 Elite），3.43 ms（8 Elite Gen 5）。
- 按计算量 9.5 倍外推：约 42 ms 和 33 ms。再乘 1.2–1.5 的「大模型更吃内存带宽」系数（系数取自上面 iPhone 的观察，不是 Android 实测）：约 **40–65 ms 每张**，前提是整个模型落在 NPU 上。
- 没找到官方先例证明 632 M 参数、1.26 GB 的 ViT 能整个落在 Snapdragon NPU 上。如果只能跑 GPU，按 Google 说的「NPU 比 GPU 最高快 10 倍」，大约 **0.3–0.6 秒**；跑 CPU 则是 **数秒**。
- 合起来：**旗舰 Android 上每张 0.05–0.6 秒量级，CPU 保底数秒**。非高通旗舰（Tensor、Dimensity）没有可比的一手 ViT 数字，不做估算。

**这些估算漏掉的东西：** 按计算量缩放不考虑内存带宽（Apple 自己说很多 transformer 配置在 ANE 上是带宽受限的）、不考虑发热降频、不考虑首次加载和编译时间（SDXL 的例子里 Apple 提到某种 attention 实现「编译时间长到不可接受」）、也不含检测器的时间。一张照片如果有多只鸟，识别模型要跑多次。

## 五、替代方案（各一行）

- **Mac / 桌面跑大模型，手机只显示结果。** 精度和速度就是现在已测的 92.4% 和 33–40 img/s，零移植风险；代价是手机离开电脑就不能识别。
- **服务器端推理。** 手机端最简单，任何机型都能用；代价是要运维和付费，照片要上传，没网不能用，和「开源优先、本地运行」的方向不一致。
- **继续蒸馏小模型。** 22 M 的 ViT-S 计算量是 ViT-B/16 的约四分之一，按上表 ViT-B 的 4–10 ms 缩放，应在几毫秒量级（估算）；代价是数据量。一手证据：DINOv2 的小模型是从 ViT-g 蒸馏的，用的是 142 M 张图的 LVD-142M；MobileCLIP 用 DataCompDR-1B（12.8 亿对）训练，见过 130 亿样本；TinyCLIP 的 ViT-8M/16 在 YFCC-15M 上训练，ImageNet 零样本 41.1%。我们的学生目前 60%，要追上 92.4% 大概率还需要量级更大的数据或更长的训练（这句是判断，不是引用）。
  来源：https://arxiv.org/abs/2304.07193 、https://arxiv.org/abs/2311.17049 、https://github.com/apple/ml-mobileclip 、https://arxiv.org/abs/2309.12314
- **int4 / palettized 权重。** 文件降到约 0.32 GB（4-bit）或 0.47 GB（6-bit）（估算），在 ANE 上还可能更省内存；代价是精度没保证。Apple 对 Stable Diffusion 的建议是不微调时最低 6-bit，更低需要微调或混合位宽方案。Core ML 的 4-bit 线性量化和 per-grouped-channel palettization 需要 iOS 18。必须用我们的 eval 集重测 top-1。
  来源：https://github.com/apple/ml-stable-diffusion 、https://apple.github.io/coremltools/docs-guides/source/opt-whats-new.html

## 没核实的地方

以下各条没有找到一手来源，或者只是推断。

1. **ViT-H/14 在任何手机上的实测延迟。** 没有。查过：Apple Core ML 模型库、ml-stable-diffusion、ml-mobileclip、MobileCLIP 论文、高通 AI Hub 的 ViT / CLIP / OWL-ViT 页面、ExecuTorch 例子目录、Google 的 LiteRT QNN 博客。文中的延迟全部是外推。
2. **632 M 参数的模型能否整个跑在 ANE 上。** 没找到 Apple 公布的 ANE 模型大小上限。「超过 1 GB 要切块」只出现在 Stable Diffusion 仓库对 UNet 的说明里，推广到我们的模型是推断。
3. **iPhone 每个 app 的内存上限（jetsam 阈值）和 increased-memory-limit 能多给多少。** Apple 文档只说「部分机型」「会变化」，不给数字。查过 entitlement 页、jetsam 报告页、`os_proc_available_memory` 页。
4. **iPhone 的 RAM。** Apple 规格页不列。没有引用任何数字。
5. **iOS 蜂窝网络下载大小限制。** 搜到的全是 2019 年的媒体报道（200 MB），以及一个开发者论坛帖子提到设置里有「超过 200 MB 时询问」的选项。没有找到 Apple 的正式文档；查过 App Store Connect 帮助的包大小页、Xcode 文档的「Reducing your app’s size」页和 iPhone 使用手册的蜂窝数据设置页。当前是否还有硬限制，未核实。
6. **Core ML 的 8-bit 权重在运行时占多少内存。** 文档说因硬件和 compute unit 而异。0.63 GB 的文件在内存里是 0.63 GB 还是展开成 1.26 GB，要真机量。
7. **增加 block 15 输出后转换是否顺利、是否影响速度。** open_clip 有 API、`ct.convert` 支持多输出都已确认，但组合起来没有实测。ExecuTorch 多输出没有找到专门文档。
8. **ExecuTorch XNNPACK 是否支持「8-bit 权重加浮点激活」的 weight-only 模式。** 文档只写「支持 8-bit 量化」。Vulkan 后端明确支持。
9. **Snapdragon NPU（HTP）对模型大小的上限。** 没有在 ExecuTorch 或 ONNX Runtime 的 QNN 文档里找到数值。
10. **ViT 做全整型量化（8a8w）会掉多少精度。** 文中说「有风险」是经验判断，没有引用来源。
11. **高通 AI Hub 上有没有 ViT-L 或更大的 ViT。** 只搜了一次，没搜到；没有逐页翻完整的模型目录。ViT-L/14 那一行的 Core ML Tools 和 iOS 版本论文没写。
12. **A14 到最新 iPhone 芯片的整体加速比。** 只有 Apple 模型库里 iPhone 13 Pro Max 到 16 Pro 的两个小模型数据点（约 1.6 倍），大模型是否同样受益未核实。
13. **Stable Diffusion UNet 的参数量。** 表中写「未列」，没有从记忆里补数字。
14. **Android native 内存的每 app 上限。** 官方内存文档只讲 Java heap 上限，没给 native 内存的数字。
15. **OWLv2 base 在手机上的任何数据。** 只找到 OWL-ViT（上一代）的高通数据。
16. **Google Tensor 和 MediaTek NPU 上的 ViT 延迟。** 只确认了后端存在，没有找到一手延迟数字。
17. **ExecuTorch `dinov2` 例子用的是哪个尺寸。** 只确认目录存在。
