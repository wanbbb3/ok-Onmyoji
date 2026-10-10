# 契灵双开任务

本任务在“战斗”列表中显示为“契灵双开”。队长循环处理建房、跨区邀请、挑战、准备和结契结果；队员处理邀请与准备。每轮先识别当前界面，再决定下一步。

## 从源码启动

先启动游戏并进入画面，再在**管理员 PowerShell**中进入本项目目录。首次运行安装依赖：

```powershell
Set-Location 'C:\Users\wanbi\Desktop\okscript\ok-Onmyoji'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

分别在两个管理员 PowerShell 窗口启动实例：

```powershell
Set-Location 'C:\Users\wanbi\Desktop\okscript\ok-Onmyoji'
.\.venv\Scripts\python.exe main_debug.py
```

```powershell
Set-Location 'C:\Users\wanbi\Desktop\okscript\ok-Onmyoji'
.\.venv\Scripts\python.exe main_debug.py --instance 2
```

也可以只启动第一个实例，再在应用的“多开助手”选择编号 2 并点击“打开新窗口”；它会使用同一个虚拟环境启动第二个实例。

每个实例在软件内选择各自的游戏窗口，并先用“截图”按钮确认截图正常。桌面专版选 Windows 游戏窗口（`onmyoji.exe` 或 `onmyoji_future.exe`），使用 WGC 截图和 PostMessage 点击；不要选普通模拟器的 ADB 设备。两个实例的任务设置分别保存在 `configs/QiLingTask.json` 和 `configs/instance_2/QiLingTask.json`。在任务设置里核对“邀请好友名”，再分别点击开始。默认开启“按窗口面积自动分配角色”：任务开始时，已选中的桌面游戏窗口中面积最大者为队长，其余为队员。先把队长窗口调大，并确认两个实例各自选中了不同的游戏窗口；如需固定角色，关闭此选项并分别设置“角色”。本地配置目录已被 Git 忽略，更新代码不会覆盖这些设置。

## 素材和上游更新

契灵专用素材放在 `assets/qiling_annotations.json`、`assets/images/qiling/` 和 `assets/qiling_reference/`。任务使用独立的特征集，原项目的 `assets/coco_annotations.json` 不需要改动。代码只在 `src/config.py` 增加一条任务注册，因此后续同步上游时冲突范围较小。

本地开发分支为 `feature/qiling-loop`，原始 `master` 保持用于跟踪 `origin/master`。当前 `origin` 指向 `https://github.com/YunLiuZ/ok-Onmyoji.git`。同步上游后，将更新重放到功能分支：

```powershell
git fetch origin
git switch master
git pull --ff-only origin master
git switch feature/qiling-loop
git rebase master
```

若以后把 `origin` 改成你自己的 GitHub fork，需要另加 `upstream` 指向原仓库，并把上述获取上游更新的命令改用 `upstream`。同步前先保存未提交的改动；配置文件和虚拟环境不会随 Git 提交。
