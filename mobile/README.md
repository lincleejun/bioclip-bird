# mobile：给手机用的模型文件

`./mobile/ct` 和 `./run` 一样借用 `../bioscan` 的环境，只是临时加上 coremltools 9.0（`uv run --with`，不改那个环境）。
产物都在 `mobile/build/`，不进 git。

| 命令 | 产物 | 体积 |
|---|---|---|
| `./mobile/ct mobile/convert.py bioclip` | `BioCLIP.mlpackage`：224×224 图 → 单位长度的 1024 维向量 `embedding` + 第 15 层 256 个小块 `tokens15`（float16）。从 16 位瘦身文件转，计算 float16，权重 8 位线性量化（逐输出通道） | 633.9 MB |
| `./mobile/ct mobile/convert.py detector` | `Detector.mlpackage`：D-FINE nano（Apache-2.0），640×640 拉伸图 → 300 个"鸟"分数 + 框（0–1 的 x0 y0 x1 y1） | 15.5 MB |
| `./mobile/ct mobile/convert.py names` | `names.f16`（11,172 × 1024 float16，行优先）+ `names.tsv`（学名、英文名、中文名、属、科；5,330 种有中文名） | 22.9 MB + 0.8 MB |
| `./mobile/ct mobile/convert.py heads` | `Heads.mlpackage`：`tokens15` → 眼嘴（heat / offset / visible）+ 飞行概率。**非商用权重，发布在另一个仓库** | 1.4 MB |

D-FINE 转换要绕过 coremltools 9.0 + torch 2.14 的三个问题（`convert.py` 里有注释）：`int()` 转形状 (1,) 的常量、位置编码里 `outer` 拿到 int32、`F.linear` 用一维权重；锚点按 640 输入固定成常量。

## 验证（Mac 上用 Python 跑 Core ML）

`./mobile/ct mobile/check.py golden --n 600`：`data/rec/golden` 里教师判为鸟的 1,549 张裁剪图里均匀取 600 张（鸟名录最高分高于兽名录最高分算鸟）。
PyTorch = 同一个 16 位权重、32 位计算、同一张 448 px 的 jpg。"录下的"= 当时教师在原裁剪图上算的向量（jpg 是事后缩小重存的，所以两边本来就到不了 1）。

| 比较 | 余弦均值 / 最低 | 鸟种第一名一致 |
|---|---|---|
| Core ML（默认算力）vs PyTorch，同一输入 | 0.99963 / 0.99586 | 592/600 |
| Core ML vs 录下的教师向量 | 0.98877 / 0.88971 | 562/600 |
| 对照：PyTorch vs 录下的教师向量 | 0.98907 / 0.88969 | 564/600 |
| Core ML 只用 CPU vs PyTorch（前 50 张） | 0.99956 / 0.99842 | 49/50 |

两个头（Core ML 头读 Core ML 的 `tokens15`，对 PyTorch 头读 PyTorch 的小块）：飞不飞 600/600 一致，概率最大差 0.0194；
眼嘴点距中位数 0.06–0.07 px（224 px 图上），3 px 以内：嘴 97.2%、左眼 99.2%、右眼 98.3%（其余是热图第一名换了格子）；"看得见"一致 99.5–100%。
量化掉的精度很小（对教师 562 vs 564），没有换量化方式。没测 6 位调色板量化。

`./mobile/ct mobile/check.py detector --n 80`：评测集 80 张鸟图，Core ML（PIL 双线性拉到 640）vs transformers 原版：两边都找到鸟 79 张，最大框 IoU 中位数 1.0，79 张 ≥ 0.9；1 张两边都没找到。

`./mobile/ct mobile/check.py expect 图...` 把 App 的整条路（Python 版）跑一遍，写 `mobile/expected/`（期望的前五、框、224 px 输入图、向量），iOS 测试读它。
现在的五张：四张评测集里 CC0 的 iNat 图 + 一张放大 8 倍的合成图（也是 CC0 原图）（测 2048 / 3072 缩放那一段；生成：
`./run -c "from PIL import Image; im=Image.open('../bioscan/data/inat/Zonotrichia_leucophrys/402656982_739214433.jpg'); im.resize((im.width*8, im.height*8), Image.BICUBIC).save('mobile/build/big-synthetic.png')"`）。
`expected.json` 里的路径相对仓库根目录（指向 `../bioscan/data/inat`）。

## 发布（给 App 下载）

Release 附件是平铺的单个文件，`.mlpackage` 是目录，所以包里每个文件单独上传，名字里 `/` 换成 `__`，App 下载后按路径还原再在手机上编译：

```
cd mobile/build && for f in BioCLIP.mlpackage Detector.mlpackage; do find $f -type f; done; echo names.f16; echo names.tsv
# 每个路径 P 上传为 ${P//\//__}，例如 BioCLIP.mlpackage__Data__com.apple.CoreML__weights__weight.bin
```

主仓库（MIT）：BioCLIP、Detector、names。`Heads.mlpackage` 的三个文件放 `lincleejun/bioclip-bird-heads` 的 Release（非商用）。
App 里的地址常量在 `ios/BirdID/Store.swift`（tag 先写的 `mobile-v0`）。

要忽略的路径：`mobile/build/`、`ios/BirdID.xcodeproj/xcuserdata/`、`ios/BirdID.xcodeproj/project.xcworkspace/xcuserdata/`。

## Android（只写路线，没建工程）

- **模型**：同一个 PyTorch wrapper（`convert.py` 的 `Tower` / `Heads` / `Detector`，输入改成 0–1 的张量）用 `torch.export` 导出。首选 ExecuTorch：XNNPACK 后端先跑通（CPU，8 位权重），再试 Qualcomm QNN / Vulkan。备选 ONNX Runtime（XNNPACK，QNN）。名录矩阵和 `names.tsv` 原样复用。
- **预处理**：照搬 `ios/BirdID/Pixels.swift`（PIL 缩放算法的移植）到 Kotlin，同样用 `mobile/expected/` 对答案。Android 的 JPEG 解码和 PIL 也会差一两个灰阶。
- **下发**：模型 0.67 GB 超过 Play base module 500 MB 上限，用 asset pack / AI pack（单个 ≤ 1.5 GB），或沿用首次启动从 Release 下载。
- **预计的坑**：D-FINE 在 coremltools 上踩到的几处（动态锚点、一维权重的 linear）在 `torch.export` 里可能同样要绕；ViT-H 8 位能不能整个上 NPU 没有先例；8 GB 内存的机器上要实测内存；8 位权重加浮点激活在 XNNPACK 上是否支持，文档没写清（见 `notes/mobile-feasibility.md`）。
