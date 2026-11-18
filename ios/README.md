# 观羽（iOS 第一版）

SwiftUI，iOS 17+，没有第三方依赖。相册选一张 → D-FINE nano 找鸟（多只时取最大的，点别的框换一只）→ BioCLIP 一遍出前五鸟种、
第 15 层小块 → 小头出眼、嘴、飞不飞 → 结果页（照片上画框和眼嘴圈，下面是名字、把握、其他可能）。每次结果自动进历史（只存在本机，可删单条、可清空）。

把握规则照抄 gardener 的 `species_level`（不含地点先验）：第一名 ≥ 50% 且领先第二名 ≥ 30 个百分点才报种；否则前五在同一属 / 科合计 ≥ 60% 时报到属 / 科；再不行就写"不确定"。

## 构建和运行

```
cd ios && xcodegen generate        # project.yml -> BirdID.xcodeproj（生成的工程也在目录里，没装 xcodegen 也能直接打开）
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild -project BirdID.xcodeproj -scheme BirdID \
  -destination 'platform=iOS Simulator,name=iPhone Duo' build      # 或 test：跑 BirdIDTests
xcrun simctl install booted <DerivedData>/Build/Products/Debug-iphonesimulator/BirdID.app
xcrun simctl launch booted cc.outman.birdid -modelDir $PWD/../mobile/build -debugImage /path/to/bird.jpg
```

- 模型：第一次打开时下载（约 0.67 GB，只走 Wi-Fi），地址常量在 `BirdID/Store.swift`，现在是占位的 Release tag。
  `-modelDir 目录` 直接从一个文件夹读（模拟器能读 Mac 上的路径，`mobile/build` 即可）。`.mlpackage` 第一次用时在手机上编译，编译结果放在 Caches。
- `-debugImage 路径`：启动后直接识别这张图，用来截图（模拟器里点不到相册选择器时用）。
- **模拟器里 Core ML 只能设 `cpuOnly`**（代码里 `#if targetEnvironment(simulator)`）：用默认的 `.all` 时 BioCLIP 两个输出全是 0（8 位版；16 位版那次也是 0，但没确认测试真的加载了它）；D-FINE 正常。Mac 上用 Python 跑同一个模型，默认算力和只用 CPU 都正常。真机上哪种算力能用、快不快，**未验证**。

## 小检查

`BirdIDTests/PipelineTests.swift`：读 `mobile/expected/`（`mobile/check.py expect` 用 Python 跑同一条路写的），对每张图比：
2048 帧尺寸、最大的框（IoU > 0.95）、送进 BioCLIP 的 224 px 图逐像素、向量余弦（> 0.995）、第一名相同、前五至少 4 个相同、飞不飞相同。
模拟器上最近一次（`xcodebuild ... test`，TEST SUCCEEDED）：

| 图（评测集 iNat，CC0） | 224 px 输入和 Python 的差 | 余弦 | 前五 |
|---|---|---|---|
| 白头海雕（飞） | 平均 0.23 灰阶，最大 17 | 0.9995 | 五个同序 |
| 褐鹈鹕（一群） | 平均 0.15，最大 3 | 0.9999 | 五个同序 |
| 西丛鸦（真值 *A. californica*；不加地点先验时两边都认成 *A. woodhouseii* 94%） | 平均 0.20，最大 3 | 0.9991 | 五个同序 |
| 雪鹭 | 平均 0.15，最大 3 | 1.0000 | 五个同序 |
| 白冠带鹀放大 8 倍（png，走 2048 / 3072 缩放） | 0（逐像素相同） | 0.9997 | 五个同序 |

另有 `testLevel`：种 / 属 / 科 / 不确定四种情况各一例。
png 完全相同，说明缩放和裁剪的移植是对的；jpg 的差来自苹果和 libjpeg 的 JPEG 解码不同，不是预处理写错。

截图（模拟器 iPhone Duo，iOS 27.1，评测集里 CC0 的 iNat 图）：`mobile/screenshots/result-eagle.png`（框、眼嘴圈、在飞）、
`result-pelican-no-heads.png`（`-modelDir` 指向不含 `Heads.mlpackage` 的目录：只出鸟种；一群鹈鹕，框很多）、`home-history.png`（历史 + "组件没装"提示）。

## 第一版没做的

- 按地区和日期修正鸟种（BirdNET geo）、稀有度、IUCN 等级：许可没定（见下）。
- 拍摄信息（相机、镜头、曝光、时间、地区）、读照片 GPS 和定位。
- 分享成图（PRD 里待定）。
- 手动框鸟（找不到鸟时现在只说"没找到鸟"）；历史记录里不能换另一只鸟（换鸟要原图还在内存里）。
- 下载没有断点续传、没有进度条（只显示第几个文件），没校验文件。HEIC 能读（系统解码），但没和电脑版对过。
- 真机上的速度、内存、首次编译时间、连续 50 张不崩：全部**未验证**。模拟器的速度不作数。

## 两个头的许可

眼嘴头（CUB-200 训练）和飞行头（iNaturalist CC BY-NC 照片训练）只限非商用，放在另一个仓库 `lincleejun/bioclip-bird-heads`，
App 里是可选组件：`Heads.mlpackage` 不在时只认鸟种，不显示眼嘴圈和飞行。主仓库（MIT）只有 BioCLIP 8 位版、D-FINE 和名录。

## 许可核对（只看了本机文件，没联网）

- **BirdNET geo 3.0**：本机说法矛盾。`birdnet` 1.1.1 包的 METADATA 和 bioscan 的文档说模型是 CC BY-NC-SA 4.0；
  但 2026-11-12 缓存的 `birdnet-team/geomodel` 仓库 README 和文档页写"代码 MIT，模型产物（checkpoint、转换后的模型、标签）Apache-2.0"，
  而 `birdnet` 包下载 geo 3.0 用的正是这个仓库的 Release（v3.0.4）。本机的模型文件旁边没有 LICENSE。
  如果 geomodel 仓库的说法是准的，geo 3.0 可以和 MIT 一起分发（Apache-2.0 要保留声明）；要主人去 Release 页面确认一次再做。
- **IUCN 等级**（`distill/rarity.py` 经 GBIF 接口取 `iucnRedListCategory`）：本机没有任何许可条款，结论"未核实"。
  `rarity.py` 自己写了"表格离开本机时要引用 GBIF 和 IUCN Red List"。IUCN 红色名录的使用条款是否允许随 App 再分发，要主人查。
- GBIF 记录数（稀有度）：本机同样没有许可文件；`data/rarity` 还在生成中（`species.csv` 只有 16 KB）。
