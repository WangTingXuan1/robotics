# 实验一：视触主动感知（电脑仿真版）

原始项目位置：`/home/maxwa/study/robotics`。使用 Conda 环境 `robotics-exp1`。

## 组员首次配置

在 Ubuntu 或 WSL 中安装 Conda 和 Git，然后克隆本仓库：

```bash
git clone <仓库地址>
cd <仓库目录>
./setup.sh
./run.sh all
```

`setup.sh` 会创建 Conda 环境、安装依赖并下载固定版本的 TACTO。需要能访问 GitHub 和 Python 包源。当前实现使用 CPU 版 PyTorch；小规模课程实验无需显卡。

## 一条命令运行

打开 Ubuntu/WSL 终端：

```bash
cd /home/maxwa/study/robotics
./run.sh all
```

这会重新生成数据、训练三个 CNN 并输出结果。现有结果已跑通，可先直接查看：

```bash
cd /home/maxwa/study/robotics
./run.sh evaluate
```

也可分步运行：

```bash
./run.sh generate 50
./run.sh train 35
./run.sh evaluate
```

如需手动进入环境：

```bash
source /home/maxwa/anaconda3/etc/profile.d/conda.sh  # 本机路径；其他电脑按 Conda 安装位置调整
conda activate robotics-exp1
cd /home/maxwa/study/robotics
export PYOPENGL_PLATFORM=egl
python experiment.py evaluate
```

## 实验内容

- PyBullet 无界面渲染视觉图像；三类物体表面分别为 smooth、ridges、bumps。
- 每个场景有三个候选接触点。代码生成局部表面深度场，再用 TACTO 渲染触觉图像。
- 分别训练视觉 CNN、触觉 CNN、视觉与触觉特征拼接的融合 CNN。
- 主动选点仅根据**训练集**各接触点的类间差异和当前类别概率选择下一点；随机选点作为对照。最多接触两次。
- 训练/验证/测试按场景分为 60%/20%/20%，不是从同一场景拆分图片。生成 150 个场景时，测试集为 30 个场景。

**范围说明：**本项目使用程序化表面深度场驱动 TACTO 渲染，展示主动选点和视触融合的完整流程。深度场并非 PyBullet 碰撞力、材料变形或真实触觉传感器测得的数据；不能把结果解释为真实材料识别性能。若课程要求“传感器和物体在物理引擎中真实碰撞”，还需要将深度场生成模块换成 TACTO 的 `Sensor` 与 URDF 接触场景。

## 输出

输出位于 `results/`：

- `dataset.npz`：图像、标签及数据划分。
- `vision.pt`、`touch.pt`、`fusion.pt`：训练好的模型。
- `metrics.json`：三类模型准确率、主动与随机选点结果。
- `comparison.png`：两组对照图。
- `demo.gif`、`demo.mp4`：一个测试样本从视觉初判、接触到最终预测的演示。

当前默认运行的测试结果：仅视觉 40.0%，仅触觉 56.7%，融合 54.4%；主动选点 96.7%，随机选点 83.3%。主动选点平均接触 1.67 次，随机选点 1.90 次。数值仅针对本项目生成的 30 个测试场景，不代表公开数据集或真实物体的性能。

## 环境说明

- Python 3.10，PyTorch 2.14.1+cpu，PyBullet 3.2.7，TACTO 0.0.3。
- TACTO 官方仓库固定为提交 `a21d0c4626d74a546d94859226c2fea348babb6a`，在 `vendor/tacto`。它由 `setup.sh` 下载，不会直接提交到本仓库。
- 官方 TACTO 依赖的旧 OpenGL Git 地址在安装时无法连接；本环境改用 PyPI 的 PyOpenGL 3.1.0 和 EGL 无界面渲染，并已实际验证。
- `urdfpy` 的元数据要求旧版 `networkx` 与 `pycollada`，但旧版 `networkx` 与 Python 3.10 不兼容。环境采用 networkx 2.8.8、pycollada 0.9.3；项目所用 TACTO 渲染接口已验证可运行。

参考：[TACTO 官方仓库](https://github.com/facebookresearch/tacto)、[Touch and Go](https://touch-and-go.github.io/)、[Touch100k](https://cocacola-lab.github.io/Touch100k/)、[ViTaPEs](https://arxiv.org/abs/2505.20032)。后三项用于了解公开数据与更复杂的融合方法，当前代码没有下载或复现它们。
