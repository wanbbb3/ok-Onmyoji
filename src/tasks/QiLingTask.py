"""契灵组队任务：队长建房、选择指定队友并邀请，队友接受邀请。

庭院「探索」和探索地图「契灵之境」使用 OCR 定位；进入契灵页面后，
先验证页面标题，再处理现有 COCO 标注。点击下一步前必须确认上一步已切换成功。
复用项目的多开实例和独立配置；每个实例选好游戏窗口后手动启动任务。
"""

import os
import re
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
try:
    from opencc import OpenCC
except ImportError:
    OpenCC = None

from ok.feature.Box import Box
from ok.feature.FeatureSet import FeatureSet
from ok.util.process import is_admin

from src.tasks.BaseOmjTask import BaseOmjTask

# 全屏搜索方差：模板按整张游戏截图标注，按钮在固定相对位置，放大到全屏避免分辨率漂移漏检。
HV = 1.0
VV = 1.0

# --------------------------------------------------------------------------- 界面
# 核心原则：**先判断当前在哪个界面，再决定做什么**，绝不"点完就算过、一路往下推"。
# 上一步没点中时界面不会变，靠顺序推进会一路点错；以画面为准才能自动回退/重试。
#
# 每个界面用"只有在这个界面上才出现的按钮"来标识（主模板 + 备份模板都列上）。
SCREEN_MARKS = {
    # --- 队长建房流程 ---
    "主界面-探索":      [],  # OCR：庭院中的「探索」灯笼
    "契灵之境入口":     [],  # OCR：探索地图底部的「契灵之境」
    "选择目标-镇墓兽":  ["3-选择镇墓兽"],
    "求援设置":         [],  # OCR：选择镇墓兽后出现的「求援」
    "创建房间":         [],  # OCR：求援弹窗中的「创建」
    "队伍空位-加号":    [],  # OCR：创建队伍后画面中部的独立「+」
    "跨区确认":         [],  # OCR：加号弹窗中的「跨区」
    "等待指定队友": [],  # OCR：跨区后等待配置的队友出现
    # --- 组队房间（队长视角：能看到"挑战"）---
    "组队房间-队长":    ["8-点击挑战", "8-队友加入"],
    # --- 队友视角 ---
    "邀请弹窗-队友":    ["队友-选择接受", "队友-选择自动接受"],
    "组队房间-队友":    ["点击准备开始"],
    # --- 通用 ---
    "过场-点击继续":    ["点击屏幕继续"],
    "结算-确定":        ["11-1确定"],
    "结算-继续邀请":    ["11-继续邀请队友"],
    "放弃结契":         ["10-放弃结契"],
}

# 队长建房的界面顺序。每轮以“实际识别到的界面”校正进度：
# 识别到更靠后的界面 -> 推进；识别到更靠前的 -> 回退（说明上一步点击没生效）。
HOST_FLOW = [
    "主界面-探索",
    "契灵之境入口",
    "选择目标-镇墓兽",
    "求援设置",
    "创建房间",
    "队伍空位-加号",
    "跨区确认",
    "等待指定队友",
]

# --------------------------------------------------------------------------- 文字识别
# 有些界面（尤其是点完某步弹出的小窗口）用图片模板很难匹配：
# 模板是拿"框内全部像素"去比，框内的背景一换（白天/夜晚、弹窗底色不同）分数就掉到阈值以下。
# 而**按钮上的文字是稳定的**，所以这些界面改用 OCR 直接认字。
# 区域取自现有截图及运行日志中的按钮位置，留有余量，
# 只限制在相应 UI 区域，不限制人物/背景的位置。
SCREEN_OCR = {
    "主界面-探索": ("探索", (0.42, 0.08, 0.66, 0.34)),
    # 探索页的底栏文字有时被 OCR 合并成「妖气探索契灵之境」，用正则找其中的目标文字。
    "契灵之境入口": (re.compile(r"契灵之境"), (0.0, 0.60, 1.0, 1.0)),
    "求援设置": ("求援", (0.55, 0.50, 1.0, 1.0)),
    "创建房间": ("创建", (0, 0, 1, 1)),
    "跨区确认": ("跨区", (0, 0, 1, 1)),
    "等待指定队友": (None, (0.10, 0.20, 0.55, 0.58)),
    "邀请队友": ("邀请", (0, 0, 1, 1)),
}
PLUS_OCR_REGION = (0.15, 0.20, 0.90, 0.90)
DEFAULT_INVITE_NAME = "过往的兔耳2"
_INVITE_T2S = OpenCC("t2s") if OpenCC is not None else None
ROOM_TEAMMATE_REGION = (0.03, 0.03, 0.83, 0.62)
HOST_OCR_STAGES = {"求援设置", "创建房间", "跨区确认", "等待指定队友"}
# 庭院-1.png 的「探索」是竖排，OCR 通常拆成「探」「索」两个框。
HOME_VERTICAL_OCR_REGION = (0.32, 0.27, 0.40, 0.44)
# 探索页底部和契灵页左上角都有「契灵之境」，须分别验证所在位置。
QILING_PAGE_OCR = ("契灵之境", (0.07, 0, 0.37, 0.15))
# 选择镇墓兽后，结契页左侧会同时显示这些固定菜单项。
# 「召唤」在运行日志中偶有漏识别，因此用「结契设置」加另外任意一项确认页面。
BINDING_PAGE_OCR = {
    "结契设置": (0.00, 0.46, 0.20, 0.80),
    "召唤": (0.20, 0.48, 0.36, 0.72),
    "释放": (0.20, 0.67, 0.36, 0.87),
}
DEFAULT_OCR_SCREENS = "主界面-探索,契灵之境入口,求援设置,创建房间"
TARGET_TO_SUPPORT_RETRY_SECONDS = 8.0
INVITE_RECHECK_SECONDS = 2.5
_QILING_TITLE_REFERENCE = None
_HOME_VERTICAL_REFERENCE = None


def _marker_matches_friend(marker, name_box):
    """只接受位于目标好友姓名右侧、同一行的选中对号。"""
    if marker is None or name_box is None:
        return False
    marker_x = marker.x + marker.width / 2
    marker_y = marker.y + marker.height / 2
    name_right = name_box.x + name_box.width
    name_y = name_box.y + name_box.height / 2
    return (name_right - name_box.height <= marker_x <= name_right + max(name_box.width, name_box.height * 5)
            and abs(marker_y - name_y) <= max(25, name_box.height * 1.5))


def _checkbox_center(box):
    """复选框在「默认邀请队友」文字左侧，以文字高度估算中心。"""
    return int(box.x - max(20, box.height * .85)), int(box.y + box.height / 2)


def _checkbox_tick_visible(frame, box):
    """勾选时深色圆圈中央出现亮色对号；只看圆圈中央以避开边框。"""
    cx, cy = _checkbox_center(box)
    radius = max(5, int(box.height * .35))
    h, w = frame.shape[:2]
    crop = frame[max(0, cy-radius):min(h, cy+radius), max(0, cx-radius):min(w, cx+radius)]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray >= 165)) / gray.size >= .10


def _checkbox_changed(before, after, box):
    """确认对号位置在点击后确实发生变化，排除原本就勾选的静态画面。"""
    if before is None or before.shape != after.shape:
        return False
    cx, cy = _checkbox_center(box)
    radius = max(10, int(box.height * .65))
    h, w = after.shape[:2]
    y0, y1 = max(0, cy-radius), min(h, cy+radius)
    x0, x1 = max(0, cx-radius), min(w, cx+radius)
    if x1 <= x0 or y1 <= y0:
        return False
    old = cv2.cvtColor(before[y0:y1, x0:x1, :3], cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(after[y0:y1, x0:x1, :3], cv2.COLOR_BGR2GRAY)
    return np.count_nonzero(cv2.absdiff(old, new) >= 35) >= max(12, int(old.size * .06))


def _reference_crop(filename, region):
    """只取截图中静态按钮/标题，排除会动的人物和背景。"""
    path = Path(__file__).resolve().parents[2] / "assets" / "qiling_reference" / filename
    if not path.is_file():
        return None
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    height, width = image.shape[:2]
    x, y, to_x, to_y = region
    crop = image[int(y * height):int(to_y * height), int(x * width):int(to_x * width)]
    return (crop, width, height) if crop.size else None


def _qiling_title_reference():
    global _QILING_TITLE_REFERENCE
    if _QILING_TITLE_REFERENCE is None:
        _QILING_TITLE_REFERENCE = _reference_crop("契灵之境.png", (0.11, 0.014, 0.25, 0.095))
    return _QILING_TITLE_REFERENCE


def _home_vertical_reference():
    global _HOME_VERTICAL_REFERENCE
    if _HOME_VERTICAL_REFERENCE is None:
        _HOME_VERTICAL_REFERENCE = _reference_crop("庭院-1.png", (0.34, 0.29, 0.39, 0.42))
    return _HOME_VERTICAL_REFERENCE


def _match_reference(frame, reference, search_region, threshold):
    """在宽松搜索区匹配截图裁出的静态小块，返回当前画面中的 Box。"""
    if frame is None or reference is None:
        return None
    crop, original_width, original_height = reference
    height, width = frame.shape[:2]
    scaled = cv2.resize(crop, (max(1, round(crop.shape[1] * width / original_width)),
                               max(1, round(crop.shape[0] * height / original_height))))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    x, y, to_x, to_y = search_region
    left, top = int(x * width), int(y * height)
    area = gray[top:int(to_y * height), left:int(to_x * width)]
    if area.shape[0] < scaled.shape[0] or area.shape[1] < scaled.shape[1]:
        return None
    _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(area, scaled, cv2.TM_CCOEFF_NORMED))
    if score < threshold:
        return None
    return Box(left + location[0], top + location[1], scaled.shape[1], scaled.shape[0],
               confidence=score)

# 识别不出界面时，隔多少秒报告一次（并列出画面上分数最高的几个特征）
UNKNOWN_REPORT_DEFAULT = 10.0
# 同一个界面上点了这么多次还没变化，就判定"点击疑似无效"并告警（之后大幅放慢重试）
MAX_RETRY_ON_SCREEN = 4
# 在同一界面上重复点击的间隔：先点一次，然后等这么久看界面有没有变化，
# 期间靠 OCR / 模板去认新出现的窗口（避免每轮狂点同一个按钮）
RETRY_INTERVAL_DEFAULT = 4.0

# 显式指定备份模板（可选）：主特征在当前画面匹配不上时，依次用这些同义截图再试一次。
# 一般不需要手写 —— 下面的自动发现会把「xxx-备份」自动挂到对应主特征上。
# 只有当备份名和主特征名对不上（例如名字里没有共同关键词）时，才在这里手工指定。
FEATURE_FALLBACKS = {}

# 备份特征名的后缀（自动发现用）。
_BACKUP_SUFFIX_RE = re.compile(r"[-_\s]*(备份|备用|备选|替补)$")
# 特征名开头的序号（1- / 11-1 / 2- ...）
_INDEX_PREFIX_RE = re.compile(r"^\d+(-\d+)*-?")
# 特征名开头的动作词，比对关键词时去掉
_ACTION_PREFIXES = ("点击选择", "点击", "选择", "前往", "进入", "打开")

# 自动发现的备份映射缓存：{主特征名: [备份特征名, ...]}
_BACKUP_MAP_CACHE = None
# 特征库快照缓存：(文件 mtime, [全部特征名])
_NAMES_CACHE = None


def _coco_path():
    """契灵专用标注独立于上游素材库，方便以后同步更新。"""
    return str(Path(__file__).resolve().parents[2] / "assets" / "qiling_annotations.json")


def _all_feature_names():
    """特征库里全部特征名（按文件 mtime 缓存，标完图重启或文件变动后自动刷新）。"""
    global _NAMES_CACHE
    path = _coco_path()
    try:
        mtime = os.path.getmtime(path) if path else None
    except Exception:
        mtime = None
    if _NAMES_CACHE is not None and _NAMES_CACHE[0] == mtime:
        return _NAMES_CACHE[1]
    names = []
    try:
        from ok.feature.FeatureSet import load_json
        if path:
            data = load_json(path)
            names = [c.get("name") for c in data.get("categories", []) if c.get("name")]
    except Exception:
        names = []
    _NAMES_CACHE = (mtime, names)
    return names


def _is_backup_name(name):
    """名字是否像备份模板（以 备份/备用/备选 结尾）。"""
    return bool(_BACKUP_SUFFIX_RE.search(name or ""))


def _core_key(name):
    """取特征名的“核心关键词”，用于把备份图挂到对应主特征上。

    去掉序号前缀、备份后缀、以及开头的动作词，只留按钮本身的名字：
      ``3-选择镇墓兽``      -> ``镇墓兽``
      ``镇墓兽-备份``        -> ``镇墓兽``
    """
    s = (name or "").strip()
    s = _INDEX_PREFIX_RE.sub("", s)
    s = _BACKUP_SUFFIX_RE.sub("", s)
    for w in _ACTION_PREFIXES:
        if s.startswith(w) and len(s) > len(w):
            s = s[len(w):]
            break
    return s.strip("-_ ")


def _discover_backup_map(all_names):
    """从特征库全部特征名里，自动推导 {主特征: [备份特征...]}。

    规则：备份名去掉后缀得到的关键词，与某个主特征的关键词相同即配对。
    关键词有歧义（多个主特征同名）时跳过并记录，宁可不挂也不挂错。
    """
    cores = {}
    ambiguous = set()
    for n in all_names:
        if _is_backup_name(n):
            continue
        c = _core_key(n)
        if not c:
            continue
        if c in cores:
            ambiguous.add(c)
        else:
            cores[c] = n

    mapping = {}
    for n in all_names:
        if not _is_backup_name(n):
            continue
        c = _core_key(n)
        if not c:
            continue
        if c in ambiguous:
            continue
        main = cores.get(c)
        if main is None or main == n:
            continue
        mapping.setdefault(main, [])
        if n not in mapping[main]:
            mapping[main].append(n)
    return mapping


def _merge_backup_map(auto_map, explicit_map):
    """显式表优先放在前面，自动发现的结果去重后追加。"""
    merged = {}
    for main, backs in (explicit_map or {}).items():
        merged[main] = list(backs)
    for main, backs in (auto_map or {}).items():
        cur = merged.setdefault(main, [])
        for b in backs:
            if b not in cur:
                cur.append(b)
    return merged


def _instance_index():
    """复用项目的 --instance/OK_INSTANCE 多开编号。"""
    from src.config import config
    return max(1, int(config.get("instance_id", 1)))


def _default_role_for(index):
    """默认角色：双开时 #1 当队长，其余当队友；单窗口（index=0）默认队长。"""
    return "队长" if index in (0, 1) else "队员"


def _role_by_window_area(selected_hwnd, windows):
    """在已发现的桌面游戏窗口中，面积最大者为队长；同面积取左上角。"""
    if not windows or selected_hwnd not in {window[0] for window in windows}:
        return None
    captain = max(windows, key=lambda window: (window[3] * window[4],
                                                -window[2], -window[1], -window[0]))
    return "队长" if selected_hwnd == captain[0] else "队员"


def _invite_name_pattern(name):
    """匹配所配置的好友名；默认名称兼容已知的 OCR 形近字。"""
    if name == DEFAULT_INVITE_NAME:
        return re.compile(r"过往的[兔免]耳[2２](?!\d)")
    return re.compile(re.escape(name) + r"(?!\d)")


def _normalize_invite_text(text):
    """将 OCR 的繁体字及全角字符转成与配置名相同的常用字形。"""
    normalized = unicodedata.normalize("NFKC", str(text))
    if _INVITE_T2S is not None:
        return _INVITE_T2S.convert(normalized)
    return normalized.replace("煙", "烟")


class QiLingTask(BaseOmjTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "契灵双开"
        self.description = "契灵双开循环：建房邀请、挑战准备、结契及自动续邀"

        # 复用项目的多开编号与独立配置目录。
        self.instance_index = _instance_index()

        self.default_config.update({
            "角色": _default_role_for(self.instance_index),
            "按窗口面积自动分配角色": True,
            "邀请好友名": DEFAULT_INVITE_NAME,
            "匹配阈值": 0.8,
            "备份阈值": 0.7,
            "未知界面报告间隔": UNKNOWN_REPORT_DEFAULT,
            "OCR识别界面": DEFAULT_OCR_SCREENS,
            "重试间隔秒数": RETRY_INTERVAL_DEFAULT,
        })
        self.config_description.update({
            "角色": "本窗口担任的角色：队长负责建房邀请，队员负责接受邀请和准备。"
                    "多开时每个实例独立保存角色；运行中修改也会生效。",
            "按窗口面积自动分配角色": "任务开始时按已打开的桌面游戏窗口像素面积分配：最大窗口为队长，"
                                  "其余为队员。关闭此项后使用上面的手动角色设置。",
            "邀请好友名": "队长在跨区邀请列表中识别并点击的队员游戏名。请填写完整名称；运行中修改后会重新识别。",
            "匹配阈值": "主模板的相似度及格线（0~1，默认 0.8）。匹配 = 在画面上找和模板最像的位置，"
                      "相似度达到这个值才算找到。画面变化大导致偶尔找不到时可降到 0.75 左右；"
                      "调太低会误点（把别的相似按钮当成目标）。",
            "备份阈值": "备份模板的及格线（默认 0.7，故意比主模板宽松）。备份图是换场景/换分辨率时"
                      "补拍的，本身相似度会低一些，所以给它更低的门槛；日志里会显示实际分数。",
            "未知界面报告间隔": "识别不出当前在哪个界面时，隔多少秒扫描一次全特征并在日志里报告"
                          "（默认 10 秒）。识别不出界面时脚本不会乱点，只等画面变化。",
            "OCR识别界面": "保留旧配置兼容；当前流程自动识别探索、契灵之境，"
                        "用结契设置及召唤/释放确认结契页面后再找求援、创建、"
                        "中下部的加号、跨区及配置的邀请好友名，通常无需修改。",
            "重试间隔秒数": "在同一个界面上点了按钮之后，隔多少秒才允许再点一次（默认 4 秒）。"
                        "先点一次，然后等这么久看界面有没有变化 —— 等待期间会持续用 OCR/模板"
                        "去认新弹出的窗口，所以不会出现“一直狂点同一个按钮”的情况。",
        })
        self.config_type.update({
            "角色": {
                "options": ["队长", "队员"],
            },
        })

    # ------------------------------------------------------------------ 配置
    def load_config(self):
        """使用项目原有的 configs/instance_N 隔离，不另建配置规则。"""
        super().load_config()
        legacy_roles = {"host": "队长", "teammate": "队员"}
        if self.config.get("角色") in legacy_roles:
            self.config["角色"] = legacy_roles[self.config["角色"]]

    def _resolve_role(self):
        """从任务配置面板读取当前角色。

        必须在 Start 之后（而非仅 on_create 时）读取：用户在 GUI 下拉框里改完配置再点
        Start，若角色在任务创建时就被固定，改动不会生效。这里每轮循环都读一次，
        运行中切换角色也能热生效。
        """
        try:
            role = self.config.get("角色")
        except Exception:
            role = None
        role = {"队长": "host", "队员": "teammate", "host": "host", "teammate": "teammate"}.get(role)
        if role is None:
            role = _default_role_for(self.instance_index)
            role = "host" if role == "队长" else "teammate"
        if role != self.role:
            if self.role is not None:
                labels = {"host": "队长", "teammate": "队员"}
                self.log_info(f"角色切换 {labels[self.role]} -> {labels[role]}（来源：任务配置面板下拉框）")
            self.role = role
        return self.role

    def _invite_target_name(self):
        name = self.config.get("邀请好友名", DEFAULT_INVITE_NAME)
        return name.strip() if isinstance(name, str) and name.strip() else DEFAULT_INVITE_NAME

    def _assign_role_by_window_area(self):
        if not self.config.get("按窗口面积自动分配角色", True):
            return
        device_config = getattr(getattr(self.executor, "device_manager", None), "config", None)
        selected = device_config.get("selected_hwnd", 0) if device_config is not None else 0
        try:
            selected = int(selected)
        except (TypeError, ValueError):
            return
        if not selected:
            return
        from ok.util.window import find_all_visible_windows, get_window_bounds
        windows = []
        for hwnd, _, exe_name, _ in find_all_visible_windows():
            if Path(exe_name or "").name.lower() not in {"onmyoji.exe", "onmyoji_future.exe"}:
                continue
            x, y, _, _, width, height, _ = get_window_bounds(hwnd)
            if width > 0 and height > 0:
                windows.append((hwnd, x, y, width, height))
        role = _role_by_window_area(selected, windows)
        if role is not None and role != self.config.get("角色"):
            self.config["角色"] = role
            self.log_info(f"按游戏窗口面积自动分配：当前实例为{role}")

    # ------------------------------------------------------------------ 模板与阈值
    def _threshold(self, is_backup=False):
        """从配置面板读匹配阈值（主模板 / 备份模板）。"""
        key = "备份阈值" if is_backup else "匹配阈值"
        fallback = 0.7 if is_backup else 0.8
        try:
            v = float(self.config.get(key))
        except Exception:
            return fallback
        if not (0.0 < v < 1.0):
            return fallback
        return v

    def _retry_interval(self):
        """同一界面重复点击的间隔秒数。"""
        try:
            v = float(self.config.get("重试间隔秒数"))
        except Exception:
            return RETRY_INTERVAL_DEFAULT
        return v if v > 0 else RETRY_INTERVAL_DEFAULT

    def _unknown_report_interval(self):
        """识别不出界面时，隔多少秒报告一次。"""
        try:
            v = float(self.config.get("未知界面报告间隔"))
        except Exception:
            return UNKNOWN_REPORT_DEFAULT
        return v if v > 0 else UNKNOWN_REPORT_DEFAULT

    def _backup_map(self):
        """主特征 -> 备份模板列表。显式表 + 自动扫描特征库，结果缓存一次。

        自动扫描让「新标一张 xxx-备份 图」无需改代码即可生效。
        注意：只在特征库文件变化时才会重建（这里按 mtime 判断），避免每轮读文件。
        """
        global _BACKUP_MAP_CACHE
        all_names = _all_feature_names()
        mtime = _NAMES_CACHE[0] if _NAMES_CACHE is not None else None

        if _BACKUP_MAP_CACHE is not None:
            cached_mtime, cached_map = _BACKUP_MAP_CACHE
            if cached_mtime == mtime:
                return cached_map

        auto = {}
        try:
            auto = _discover_backup_map(all_names)
        except Exception as e:
            self.log_warning(f"自动扫描备份模板失败（将只用手写配置）: {e}")

        merged = _merge_backup_map(auto, FEATURE_FALLBACKS)
        _BACKUP_MAP_CACHE = (mtime, merged)
        if merged:
            detail = "；".join(f"{m} -> {'/'.join(b)}" for m, b in merged.items())
            self.log_info(f"备份模板已就绪（自动发现 + 手写配置）：{detail}")
        else:
            self.log_info("未发现任何备份模板（特征库里没有以「-备份」结尾的标注）")
        return merged

    def _templates_for(self, feat):
        """某个步骤要尝试的模板列表：[主模板] + 备份模板。"""
        return [feat] + list((self._backup_map().get(feat) or []))

    # ------------------------------------------------------------------ 生命周期
    def on_create(self):
        # 从每实例任务配置读取启动器自动写入的角色；运行中仍可在面板调整。
        self.role = None
        self.setup_step = 0                # 队长：建房流程进行到第几步（每轮以画面校正）
        self.auto_accept_enabled = False   # 队友：是否已开启自动接受
        # 界面识别状态：当前界面、进入该界面的时刻、在该界面上已点击几次
        self._last_screen = None
        self._screen_since = time.time()
        self._screen_clicks = 0
        self._unknown_since = time.time()
        self._last_unknown_report = 0.0
        self._pending_transition = None
        self._pending_last_report = 0.0
        self._host_invite_sent = False
        self._invite_phase = None
        self._invite_name_box = None
        self._active_invite_name = self._invite_target_name()
        self._invite_last_report = 0.0
        self._invite_action_at = 0.0
        self._host_phase = None
        self._room_joined_frames = 0
        self._default_invite_before = None
        self._default_invite_box = None
        self._default_checked_frames = 0
        self._last_continue_click = 0.0
        self._last_auto_accept_click = 0.0
        self._last_teammate_ready_click = 0.0
        self._last_phase_report = 0.0
        self._phase_clicked_at = 0.0
        self._resolve_role()
        cfg_file = getattr(self.config, "config_file", None) or f"QiLingTask.json"
        self.log_info(f"契灵任务就绪：实例 #{self.instance_index} 角色={'队长' if self.role == 'host' else '队员'}"
                      f" 配置文件={cfg_file}"
                      f"（可在任务设置修改后点击开始）")

    def run(self):
        # 阴阳师以管理员权限运行，脚本非管理员会被 Windows UIPI 拒绝 PostMessage。
        if not is_admin():
            self.log_error(
                "脚本未以管理员身份运行：Windows 会拒绝向阴阳师投递后台点击"
                "（PostMessage 错误 5：拒绝访问）。请在管理员 PowerShell 中"
                "从项目目录启动 main_debug.py。",
                notify=True,
            )
            return
        self._assign_role_by_window_area()
        self.log_info("契灵任务开始运行（角色每轮按配置面板刷新）")
        rounds = 0
        self._last_step = -1
        self._stall_since = time.time()
        self._last_dump = 0.0
        while not self.exit_is_set():
            rounds += 1
            try:
                # 关键：必须自己取帧。执行器只在“每个循环周期开头”采集一次画面，
                # 而本任务 run() 是常驻循环、不返回，执行器没机会再刷新 ——
                # 不主动 next_frame() 就会永远拿同一张旧图（甚至空帧）去匹配，
                # 表现为“一直在跑、没动作、也没日志”（空帧时 find_feature 静默返回空）。
                frame = self.next_frame()
                if frame is None:
                    self.log_warning("未取到游戏画面（截图失败），1 秒后重试")
                    self.sleep(1.0)
                    continue
                # 每轮都读一次配置面板：打开 GUI 后在下拉框改了角色，
                # 点 Start 会按新角色跑，运行中改动下一轮也立即生效。
                self._resolve_role()
                invite_name = self._invite_target_name()
                if invite_name != self._active_invite_name:
                    self.log_info(f"邀请好友名已改为「{invite_name}」，重新识别当前邀请步骤")
                    self._active_invite_name = invite_name
                    self._invite_phase = None
                    self._invite_name_box = None
                    self._pending_transition = None
                if rounds == 1:
                    # 启动首轮先把画面上所有已标注按钮扫一遍：一眼看出当前在哪个界面。
                    self._dump_all_scores(frame, reason="启动首轮扫描（当前画面上有哪些已标注的按钮）")
                if self.role == "host":
                    self._host_step(frame)
                else:
                    self._teammate_step(frame)
                # 卡住检测：长时间停在同一界面 / 长时间识别不出界面时，扫描画面确认状态。
                self._check_stall(frame, screen=self._last_screen)
                # 心跳日志：每 10 轮一条，确认循环在跑、画面正常、当前卡在哪一步。
                if rounds % 10 == 0:
                    self._heartbeat(frame, rounds)
            except Exception as e:  # 单轮异常不应打断整个循环
                self.log_error(f"契灵循环异常: {e}", e)
            self.sleep(1.0)

    # ------------------------------------------------------------------ 匹配诊断
    def _feature_set(self):
        """只加载契灵专用特征，避免改动上游 coco_annotations.json。"""
        fs = getattr(self, "_qiling_feature_set", None)
        if fs is None:
            fs = FeatureSet(debug=False, coco_json=_coco_path(),
                            default_horizontal_variance=0.002,
                            default_vertical_variance=0.002,
                            default_threshold=0.8)
            self._qiling_feature_set = fs
        return fs

    def find_one(self, feature_name=None, horizontal_variance=0, vertical_variance=0,
                 threshold=0, frame=None, **kwargs):
        """契灵标注走独立素材库，其他项目特征仍走原有匹配器。"""
        if isinstance(feature_name, str) and feature_name in _all_feature_names():
            source = frame if frame is not None else self.frame
            if source is None:
                return None
            boxes = self._feature_set().find_one_feature(
                source, feature_name, horizontal_variance=horizontal_variance,
                vertical_variance=vertical_variance, threshold=threshold,
                limit=1, **kwargs)
            return boxes[0] if boxes else None
        return super().find_one(feature_name, horizontal_variance, vertical_variance,
                                threshold=threshold, frame=frame, **kwargs)

    def _best_score(self, name):
        """返回该特征在当前画面上的**最高相似度**（不管有没有过阈值）。

        模板匹配不是"有/没有"的布尔判断，而是算一个 0~1 的相似度分数再和阈值比。
        卡在某个按钮时，最有用的信息就是"实际拿到多少分"：
        - 0.75 左右（差一点）：阈值调低或标得更紧就能过；
        - 0.3 以下（完全不像）：说明标注的图和当前画面根本不是同一个东西
          （界面不对 / 分辨率差太多 / 标错位置），调阈值没用。

        返回 None 表示算不出来（没画面或拿不到特征集），不要当成 0 分。
        """
        try:
            fs = self._feature_set()
            if fs is None:
                return None
            frame = getattr(self, "frame", None)
            if frame is None:
                frame = getattr(getattr(self, "executor", None), "frame", None)
            if frame is None:
                return None
            # 阈值压到极低，让框架一定把最高分返回来（不做命中判定）
            boxes = fs.find_one_feature(frame, name, horizontal_variance=HV,
                                        vertical_variance=VV, threshold=0.02, limit=1)
            if not boxes:
                return 0.0
            return round(float(boxes[0].confidence), 3)
        except Exception as e:
            self.log_warning(f"计算「{name}」匹配分数失败: {e}")
            return None

    def _dump_all_scores(self, frame=None, top=10, reason=""):
        """扫描当前画面上**所有**特征各能拿多少分，按分数降序打印。

        这是排查“卡住不动”最有用的一招：很多时候并不是目标按钮匹配不上，
        而是**当前界面根本不是预期的那个页面**（上一步没点成功 / 被弹窗挡住 /
        界面还没加载完），此时目标按钮当然永远搜不到。
        扫一遍就能看到画面上到底出现了哪些已标注的东西。
        """
        if frame is None:
            frame = getattr(self, "frame", None)
        if frame is None:
            frame = getattr(getattr(self, "executor", None), "frame", None)
        if frame is None:
            self.log_warning("想扫描画面，但没有可用的截图")
            return
        names = _all_feature_names()
        if not names:
            self.log_warning("特征库为空，无法扫描")
            return
        scored = []
        for n in names:
            s = self._best_score(n)
            if s is not None:
                scored.append((s, n))
        scored.sort(reverse=True)
        th = self._threshold()
        self.log_info(f"{reason or '当前画面全特征扫描'}：共 {len(scored)} 个特征，"
                      f"分数最高的 {min(top, len(scored))} 个 ——")
        for s, n in scored[:top]:
            self.log_info(f"    {s:.3f}  {n}{'   <= 达到阈值' if s >= th else ''}")
        if scored and scored[0][0] < th:
            self.log_info("    注意：没有任何特征达到阈值，说明当前画面不在任何一个已标注的界面上"
                          "（被弹窗挡住 / 上一步没点成功 / 界面未加载完 / 分辨率差异过大）")

    def _check_stall(self, frame, screen=None):
        """长时间停在同一界面时告警并扫描画面（确认是不是界面不对 / 点击没生效）。"""
        now = time.time()
        since = getattr(self, "_screen_since", now)
        if now - since <= 60:
            return
        if now - getattr(self, "_last_dump", 0) < 90:
            return
        self._last_dump = now
        if screen:
            extra = ""
            if self.role == "host" and screen in HOST_FLOW:
                extra = f"（建房第 {HOST_FLOW.index(screen) + 1}/{len(HOST_FLOW)} 步，" \
                        f"已在此界面点击 {getattr(self, '_screen_clicks', 0)} 次）"
            self.log_warning(f"已停在界面「{screen}」{now - since:.0f} 秒没变化{extra}")
        else:
            self.log_warning("长时间识别不出当前界面")
        self._dump_all_scores(frame, top=8, reason="卡住超时，扫描当前画面确认界面状态")

    def _waiting_targets(self):
        """当前阶段在等哪些按钮，返回 [(特征名, 是否备份模板), ...]（用于报分数）。"""
        if self.role == "host" and self._host_phase in ("waiting_teammate", "waiting_auto_teammate", "waiting_ready"):
            return [("挑战", False)]
        if self.role == "host" and (self._host_phase is not None
                                    or self.setup_step >= HOST_FLOW.index("求援设置")):
            return []  # 其它后半段界面仍由 OCR 判断。
        screen = getattr(self, "_last_screen", None)
        if screen in self._ocr_screens():
            return []  # 该界面用 OCR 认字，没有模板分数可报
        marks = SCREEN_MARKS.get(screen) if screen else None
        if marks:
            available = set(_all_feature_names())
            return [(f, _is_backup_name(f)) for f in marks if f in available]
        # 还没识别到界面时，按角色报最可能在等的按钮
        if self.role == "host":
            return [("3-选择镇墓兽", False)]
        return [("队友-选择自动接受", False), ("队友-选择接受", False)]

    def _heartbeat(self, frame, rounds):
        """周期性状态日志：没匹配到按钮时也不会彻底静默，便于排查。"""
        try:
            h, w = frame.shape[:2]
            size = f"{w}x{h}"
        except Exception:
            size = "未知"
        screen = getattr(self, "_last_screen", None)
        if screen:
            stayed = time.time() - getattr(self, "_screen_since", time.time())
            how = "用 OCR 认字定位" if screen in self._ocr_screens() else "用图片模板定位"
            if self.role == "host" and screen in HOST_FLOW:
                state = f"界面「{screen}」（建房第 {HOST_FLOW.index(screen) + 1}/{len(HOST_FLOW)} 步，" \
                        f"已停留 {stayed:.0f}s，点击 {getattr(self, '_screen_clicks', 0)} 次，{how}）"
            else:
                state = f"界面「{screen}」（已停留 {stayed:.0f}s，{how}）"
        else:
            state = "尚未识别出界面（等待画面变化 / 战斗中 / 未标注的弹窗）"
        self.log_info(f"运行中（第 {rounds} 轮）：角色={'队长' if self.role == 'host' else '队员'} 画面={size} 状态：{state}")

        # 分数诊断：卡住时最关键的是"差多少分"，而不是干巴巴一句"没找到"。
        parts = []
        for name, is_backup in self._waiting_targets():
            s = self._best_score(name)
            tag = "备份" if is_backup else "主"
            parts.append(f"{name}[{tag}]={'算不出' if s is None else s}(需{self._threshold(is_backup)})")
        if parts:
            self.log_info("  匹配分数：" + "，".join(parts))

    def _ocr_screens(self):
        """当前启用文字识别的界面名列表（来自配置面板）。

        防误判：已经认出界面时，只认"流程走到附近"的界面，避免在早期界面上
        误命中同样的文字（例如主界面上也有"创建"两个字就被当成创建房间的窗口）。
        而**识别不出界面时（卡住）放开全部**——这正是"点完某一步弹出一个没标注过的
        小窗口"的场景，此时正是要靠认字把它认出来。
        """
        try:
            raw = (self.config.get("OCR识别界面") or "").strip()
        except Exception:
            raw = DEFAULT_OCR_SCREENS
        # 前两步固定使用 OCR；旧配置里只有「创建房间」也不应关闭它们。
        screens = ["主界面-探索", "契灵之境入口"]
        screens.extend(s.strip() for s in raw.split(",")
                       if s.strip() in SCREEN_OCR and s.strip() not in screens)
        pending = getattr(self, "_pending_transition", None)
        waiting_for_support = pending is not None and pending[0] == "选择目标-镇墓兽"
        # 旧配置可能没有「求援设置」；点完镇墓兽后仍须主动等待它出现。
        if self.role == "host" and (waiting_for_support or self.setup_step >= HOST_FLOW.index("求援设置")):
            if "求援设置" not in screens:
                screens.append("求援设置")
            screens.remove("求援设置")
            if pending is not None and pending[0] == "求援设置":
                # 求援弹窗可能覆盖在旧页上；先认新弹窗的「创建」，再认底下的「求援」。
                screens.append("求援设置")
            else:
                # 选择镇墓兽后优先确认「求援」，避免把同页上的「创建」当成下一步。
                screens.insert(2, "求援设置")
        else:
            screens = [s for s in screens if s != "求援设置"]
        if self.role == "host" and getattr(self, "_last_screen", None) is not None:
            # 只提前一阶段认下一步；求援弹窗叠在旧页上，仍须优先识别新文字。
            screens = [s for s in screens
                       if s not in HOST_FLOW or self.setup_step >= HOST_FLOW.index(s) - 1]
        return screens

    def _detect_ocr(self, screen, frame=None):
        """用 OCR 认字来定位界面，返回 (界面名, 描述, box, 1.0) 或 None。"""
        spec = QILING_PAGE_OCR if screen == "契灵之境页面" else SCREEN_OCR.get(screen)
        if not spec:
            return None
        word, region = spec
        if screen == "等待指定队友":
            name = self._invite_target_name()
            box = self._read_ocr(name, region, frame, exact=False, normalize_traditional=True)
            if box is None:
                return None
            text = getattr(box, "name", None) or getattr(box, "text", "")
            return screen, f"OCR:{text}", box, 1.0
        try:
            boxes = self.ocr(*region, match=word, frame=frame, log=False)
        except Exception as e:
            self.log_warning(f"OCR 识别「{screen}」失败: {e}")
            return None
        if not boxes:
            return None
        if screen == "契灵之境入口":
            boxes = [box for box in boxes
                     if (getattr(box, "name", None) or getattr(box, "text", "")).endswith("契灵之境")
                     and (frame is None or box.y >= frame.shape[0] * 0.82)]
            if not boxes:
                return None
        b = boxes[0]
        text = getattr(b, "name", None) or getattr(b, "text", "")
        if screen == "契灵之境入口":
            # OCR 会把底栏几个相邻入口并成一个宽框；不能点击合并框的中心。
            # 「契灵之境」在文字末尾，按四个汉字的高度估算其实际点击区域。
            width = min(b.width, max(1, round(b.height * 4)))
            b = Box(b.x + b.width - width, b.y, width, b.height,
                    confidence=getattr(b, "confidence", 1.0), name="契灵之境")
        if len(boxes) > 1:
            self.log_info(f"OCR 在「{screen}」上找到 {len(boxes)} 处“{word}”，"
                          f"取最靠上的：{text} @ x={getattr(b, 'x', '?')},y={getattr(b, 'y', '?')}")
        return screen, f"OCR:{text}", b, 1.0

    def _read_ocr(self, word, region, frame, exact=True, normalize_traditional=False):
        """读取指定区域里的按钮文字，过滤「自动挑战」等包含目标字的背景文案。"""
        match = None if normalize_traditional else (word if exact else re.compile(re.escape(word)))
        pattern = (_invite_name_pattern(_normalize_invite_text(word))
                   if normalize_traditional else None)
        for grayscale in (False, True):
            try:
                # 当前 onnxocr 检测器要求三通道图；use_grayscale=True 会传入二维图并抛错。
                source = (cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                          if grayscale and frame is not None else frame)
                boxes = self.ocr(*region, match=match, frame=source, log=False)
            except Exception as e:
                self.log_warning(f"OCR 识别「{word}」失败: {e}")
                return None
            for box in boxes or ():
                value = (getattr(box, "name", None) or getattr(box, "text", "")).strip()
                if normalize_traditional and pattern.search(_normalize_invite_text(value)):
                    return box
                if not normalize_traditional and (value == word or (not exact and word in value)):
                    return box
        return None

    def _room_teammate_visible(self, frame, skip_popup_check=False):
        """邀请弹窗关闭后，以房间中出现配置的好友名确认队员入房。"""
        if not skip_popup_check and self._invite_popup_visible(frame):
            return False
        return self._read_ocr(self._invite_target_name(), ROOM_TEAMMATE_REGION, frame,
                              exact=False, normalize_traditional=True) is not None

    def _room_challenge_box(self, frame):
        """用新标注的图片模板定位房间右下角的挑战按钮。"""
        challenge = self.find_one("挑战", HV, VV, threshold=self._threshold(), frame=frame)
        if challenge is None:
            return None
        h, w = frame.shape[:2]
        if challenge.x + challenge.width / 2 < w * .75 or challenge.y + challenge.height / 2 < h * .70:
            return None
        return challenge

    def _resume_room_if_ready(self, frame):
        """续邀弹窗被跳过时，以队友和挑战按钮确认已直接回房。"""
        if not self._room_teammate_visible(frame, skip_popup_check=True):
            return False
        if self._room_challenge_box(frame) is None:
            return False
        self._host_phase = "waiting_auto_teammate"
        self._room_joined_frames = 1
        self._phase_clicked_at = time.time()
        self.log_info(f"已识别到房间中的「{self._invite_target_name()}」和「挑战」，直接进入开战流程")
        return True

    def _detect_plus(self, frame):
        """只找画面中下部的独立加号，排除顶部资源栏的充值加号。"""
        for grayscale in (True, False):
            try:
                source = (cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                          if grayscale and frame is not None else frame)
                boxes = self.ocr(*PLUS_OCR_REGION, match="+", frame=source, log=False)
            except Exception as e:
                self.log_warning(f"OCR 识别队伍空位「+」失败: {e}")
                continue
            if boxes:
                box = boxes[0]
                return "队伍空位-加号", "OCR:+", box, getattr(box, "confidence", 1.0)
        return None

    def _detect_host_stage(self, screen, frame):
        if screen == "队伍空位-加号":
            return self._detect_plus(frame)
        if screen in HOST_OCR_STAGES:
            return self._detect_ocr(screen, frame)
        return None

    def _detect_host_screen(self, frame):
        """建房后半段只认当前目标，避免背景中的旧按钮或资源栏抢先命中。"""
        pending = self._pending_transition
        if pending is not None:
            previous, expected, _ = pending
            target = expected[0]
            if target in HOST_OCR_STAGES or target == "队伍空位-加号":
                found = self._detect_host_stage(target, frame)
                if found is not None:
                    return found
                if previous in HOST_OCR_STAGES or previous == "队伍空位-加号":
                    found = self._detect_host_stage(previous, frame)
                    if found is not None:
                        return found
                elif previous == "选择目标-镇墓兽":
                    return self._detect_screen(frame)
                return None, None, None, None
        if self.setup_step >= HOST_FLOW.index("求援设置"):
            target = HOST_FLOW[min(self.setup_step, len(HOST_FLOW) - 1)]
            return self._detect_host_stage(target, frame) or (None, None, None, None)
        return self._detect_screen(frame)

    def _detect_home_explore(self, frame):
        """兼容两张庭院截图中横排和竖排的「探索」。"""
        horizontal = self._detect_ocr("主界面-探索", frame)
        if horizontal is not None:
            return horizontal
        try:
            boxes = self.ocr(*HOME_VERTICAL_OCR_REGION, frame=frame, log=False)
        except Exception as e:
            self.log_warning(f"竖排「探索」OCR 失败: {e}")
            boxes = []
        for box in boxes:
            if getattr(box, "text", "").strip() == "探索":
                return "主界面-探索", "OCR:竖排探索", box, box.confidence
        glyphs = [box for box in boxes if getattr(box, "text", "").strip() in ("探", "索")]
        for first, second in zip(glyphs, glyphs[1:]):
            if (getattr(first, "text", "").strip() != "探" or
                    getattr(second, "text", "").strip() != "索"):
                continue
            first_center = first.x + first.width / 2
            second_center = second.x + second.width / 2
            if abs(first_center - second_center) > max(first.width, second.width):
                continue
            if not (0 <= second.y - first.y <= max(first.height, second.height) * 2):
                continue
            x = min(first.x, second.x)
            y = min(first.y, second.y)
            right = max(first.x + first.width, second.x + second.width)
            bottom = max(first.y + first.height, second.y + second.height)
            button = Box(x, y, right - x, bottom - y,
                         confidence=min(first.confidence, second.confidence), name="探索")
            return "主界面-探索", "OCR:竖排探索", button, button.confidence
        fallback = _match_reference(frame, _home_vertical_reference(),
                                    (0.25, 0.20, 0.50, 0.52), 0.78)
        if fallback is not None:
            fallback.name = "探索"
            return "主界面-探索", "图片:庭院-1探索", fallback, fallback.confidence
        return None

    def _qiling_page_visible(self, frame):
        """OCR 或静态标题截图确认已进入契灵之境页面。"""
        if self._detect_ocr("契灵之境页面", frame) is not None:
            return True
        return _match_reference(frame, _qiling_title_reference(),
                                (0, 0, 0.40, 0.17), 0.75) is not None

    def _binding_page_visible(self, frame):
        """通过结契页左侧固定菜单确认已进入契灵结契。"""
        found = set()
        for word, region in BINDING_PAGE_OCR.items():
            try:
                if self.ocr(*region, match=word, frame=frame, log=False):
                    found.add(word)
            except Exception as e:
                self.log_warning(f"OCR 识别结契页「{word}」失败: {e}")
        return "结契设置" in found and bool(found & {"召唤", "释放"})

    def _promote_binding_page(self, frame):
        """结契页已经出现时，跳过旧的镇墓兽等待，直接寻找求援。"""
        support_step = HOST_FLOW.index("求援设置")
        if self.setup_step >= support_step or not self._binding_page_visible(frame):
            return
        self.setup_step = support_step
        self._pending_transition = None
        self._pending_last_report = 0.0
        self.log_info("已由「结契设置」及「召唤/释放」确认进入契灵结契页面，开始识别「求援」")

    def _detect_screen(self, frame=None):
        """判断当前画面在哪个界面（这是每轮决策的第一步）。

        先跑**文字识别**（对配置了 OCR 的界面，如「创建房间」认"创建"两个字），
        再用图片模板遍历 ``SCREEN_MARKS``，取**分数最高的那个命中界面**作为当前界面。

        返回 ``(界面名, 命中的特征名/文字, box, 分数)``；都没命中时界面名为 None。
        """
        # 1) 文字识别优先：弹窗/换背景时图片模板容易失配，而按钮文字始终稳定。
        for screen in self._ocr_screens():
            got = (self._detect_home_explore(frame) if screen == "主界面-探索"
                   else self._detect_ocr(screen, frame))
            if got is not None:
                return got
        # 2) 图片模板识别其余界面
        best = (None, None, None, -1.0)
        available = set(_all_feature_names())
        for screen, marks in SCREEN_MARKS.items():
            for feat in marks:
                if feat not in available:
                    continue
                box = self.find_one(feat, HV, VV,
                                    threshold=self._threshold(is_backup=_is_backup_name(feat)))
                if box:
                    score = float(getattr(box, "confidence", 0.0))
                    if score > best[3]:
                        best = (screen, feat, box, score)
                    break  # 该界面已命中，不必再试它的备份模板
        if best[0] is None:
            # 没有后续按钮时仍能确认已经进入契灵页面，避免误认为还在探索地图。
            if self._qiling_page_visible(frame):
                return "契灵之境页面", "页面标题", None, 1.0
            return None, None, None, None
        return best

    def _dump_ocr_texts(self, frame, reason="", top=30):
        """**实时**对当前画面做一次全屏 OCR，把识别到的文字和位置打进日志。

        这是定位"某个字到底在画面哪里 / OCR 把它认成了什么"最直接的手段：
        比拿标注原图去试更可靠（运行时分辨率、场景、弹窗都和原图不同）。
        """
        if frame is None:
            return
        try:
            boxes = self.ocr(x=0, y=0, to_x=1, to_y=1, frame=frame, log=False)
        except Exception as e:
            self.log_warning(f"OCR 全文扫描失败: {e}")
            return
        if not boxes:
            self.log_info(f"{reason}：全画面 OCR 没识别到任何文字")
            return
        try:
            h, w = frame.shape[:2]
        except Exception:
            h = w = 0
        self.log_info(f"{reason}：实时 OCR 全画面识别到 {len(boxes)} 处文字 ——")
        for b in boxes[:top]:
            text = getattr(b, "text", "")
            x = getattr(b, "x", 0)
            y = getattr(b, "y", 0)
            rel = f"（相对 {x / w:.2f},{y / h:.2f}）" if w and h else ""
            mark = "   <= 含「契灵」" if "契灵" in text else ""
            self.log_info(f"    「{text}」@ x={x},y={y} {rel}{mark}")

    def _on_unknown_screen(self, frame):
        """识别不出界面：不乱点，只等待，并周期性扫描报告画面上有什么。

        未知界面通常意味着：页面正在切换（过场动画/加载中）、被未标注的弹窗挡住、
        或当前在一个还没标注过的界面（如战斗中、结算中）。
        """
        now = time.time()
        if now - getattr(self, "_last_unknown_report", 0.0) < self._unknown_report_interval():
            return
        self._last_unknown_report = now
        secs = now - getattr(self, "_unknown_since", now)
        self.log_info(f"当前画面识别不出界面（已 {secs:.0f} 秒），不点击、等待画面变化")
        self._dump_all_scores(frame, top=6, reason="未知界面，扫描画面上有哪些已标注的按钮")
        # 实时 OCR 全文：直接看画面上到底有哪些字、在哪个位置
        self._dump_ocr_texts(frame, reason="未知界面，实时 OCR 全文")

    def _note_screen(self, screen):
        """记录界面停留情况；界面变化时重置点击计数。"""
        now = time.time()
        if screen != getattr(self, "_last_screen", None):
            prev = getattr(self, "_last_screen", None)
            self._last_screen = screen
            self._screen_since = now
            self._screen_clicks = 0
            self._last_click_time = 0.0
            self._unknown_since = now
            self._last_unknown_report = 0.0
            if prev is not None:
                self.log_info(f"界面切换：{prev} -> {screen}")
            else:
                self.log_info(f"当前界面：{screen}")
        return now - getattr(self, "_screen_since", now)

    # ------------------------------------------------------------------ 队长
    def _host_scene_action(self, frame):
        """每张新截图先按实际画面分流；阶段只负责防止同一按钮被连续点击。"""
        phase = self._host_phase
        if self._binding_success_visible(frame):
            if phase is not None or self._invite_phase is not None:
                self._reset_after_success()
                return True
            if self.setup_step < HOST_FLOW.index("求援设置"):
                self.setup_step = HOST_FLOW.index("求援设置")
                self._pending_transition = None
                self.log_info("识别到镇墓兽结契页和求援按钮，从求援节点继续")
            return False

        label = self._read_ocr("默认邀请队友", (0.10, 0.15, 0.90, 0.85), frame, exact=False)
        if label is not None:
            if phase == "waiting_continue" and time.time() - self._phase_clicked_at < 1.5:
                return True
            if phase not in ("checking_default_invite", "confirm_reinvite"):
                self._host_phase = "waiting_reinvite_dialog"
                if phase != "waiting_reinvite_dialog":
                    self.log_info("识别到默认邀请队友弹窗，切换到续邀节点")
            self._host_cycle_step(frame)
            return True

        abandon = self._read_ocr("放弃结契", (0, 0, 1, 1), frame, exact=False)
        if abandon is not None:
            if phase != "waiting_continue" or time.time() - self._phase_clicked_at >= self._retry_interval():
                if self._click("放弃结契", abandon):
                    self._host_phase = "waiting_continue"
                    self._phase_clicked_at = time.time()
                    self.log_info("识别到结契失败，已点击放弃结契，等待下一画面")
            return True

        challenge = self._room_challenge_box(frame)
        if challenge is not None and self._room_teammate_visible(frame):
            self._invite_phase = None
            self._pending_transition = None
            if phase == "waiting_ready":
                if time.time() - self._phase_clicked_at >= self._retry_interval():
                    if self._click("挑战重试", challenge):
                        self._phase_clicked_at = time.time()
                return True
            if phase not in ("waiting_teammate", "waiting_auto_teammate"):
                self._host_phase = "waiting_auto_teammate"
                self._room_joined_frames = 1
                self.log_info(f"识别到房间队员「{self._invite_target_name()}」和「挑战」，进入挑战节点")
                return True
            self._host_cycle_step(frame)
            return True

        continue_box = self._read_ocr("点击屏幕继续", (0.05, 0.60, 0.95, 1), frame, exact=False)
        if continue_box is not None:
            elapsed = time.time() - getattr(self, "_phase_clicked_at", 0.0)
            if phase == "waiting_continue" and elapsed < 1.5:
                return True
            if phase != "waiting_reinvite_dialog" or elapsed >= self._retry_interval():
                if self._click("点击屏幕继续", continue_box):
                    self._host_phase = "waiting_reinvite_dialog"
                    self._phase_clicked_at = time.time()
                    self.log_info("已点击继续，等待续邀弹窗或队友直接回房")
            return True

        ready = (self._read_ocr("准备", (0.45, 0.55, 1, 1), frame)
                 or self._read_ocr("准备开始", (0.45, 0.55, 1, 1), frame))
        if ready is not None:
            if phase != "battle" or time.time() - self._phase_clicked_at >= self._retry_interval():
                if self._click("准备", ready):
                    self._host_phase = "battle"
                    self._phase_clicked_at = time.time()
                    self.log_info("识别到准备按钮，已点击并等待战斗结果")
            return True
        return False

    def _host_step(self, frame):
        # 每轮先识别组队、战斗和结算节点，任何旧阶段都不能遮挡真实画面。
        if self._host_scene_action(frame):
            return
        if self._host_phase is not None:
            self._host_cycle_step(frame)
            return
        if self._invite_phase is not None:
            self._invite_step(frame)
            return
        self._promote_binding_page(frame)
        screen, feat, box, score = self._detect_host_screen(frame)
        if not self._verify_pending_transition(frame, screen):
            return
        if screen is None:
            self._on_unknown_screen(frame)
            return
        stayed = self._note_screen(screen)

        if screen == "等待指定队友":
            self._invite_name_box = box
            if self._selected_friend_marker(box):
                self._invite_phase = "invite"
                self._invite_last_report = time.time()
                self._invite_action_at = time.time()
                self.log_info(f"标注「成功点击邀请」已出现在「{self._invite_target_name()}」右侧，确认已选中，准备点击邀请")
                return
            if self._click(self._invite_target_name(), box):
                self._invite_phase = "confirm_selection"
                self._invite_last_report = time.time()
                self._invite_action_at = time.time()
                self.log_info(f"已点击「{self._invite_target_name()}」，等待其右侧出现标注「成功点击邀请」后再点击邀请")
            return

        if screen == "契灵之境页面":
            # 页面已进入，但只有实际检测到「镇墓兽」按钮才继续下一步。
            return
        if screen == "选择目标-镇墓兽" and not self._qiling_page_visible(frame):
            self.log_warning("识别到镇墓兽按钮，但未确认契灵之境页面标题，暂不点击")
            return

        # 1) 结算弹窗（打完一把后出现，通常只有一次，这里见到就点）。
        if screen in ("结算-确定", "结算-继续邀请"):
            self.log_info(f"队长：结算弹窗「{screen}」，点击 {feat}")
            self._click(feat, box)
            return
        # 2) 过场。
        if screen == "过场-点击继续":
            self._click(feat, box)
            return
        # 3) 中途启动在战斗结果页时也按续邀流程走。
        if screen == "放弃结契":
            if self._click(feat, box):
                self._host_phase = "waiting_continue"
                self._phase_clicked_at = time.time()
            return
        # 4) 建房流程中的某个界面 —— 以画面为准校正进度后再点。
        if screen in HOST_FLOW:
            self._host_build(screen, feat, box, score, frame)
            return
        # 5) 识别到了别的界面：不动，只报告。
        if stayed > 15:
            self.log_info(f"队长当前在「{screen}」（非建房流程界面），无需操作")

    def _selected_friend_marker(self, name_box):
        """用素材库中的选中对号确认目标好友已选中。"""
        marker = self.find_one("成功点击邀请", HV, VV, threshold=self._threshold())
        return marker if _marker_matches_friend(marker, name_box) else None

    def _invite_popup_visible(self, frame):
        """区分邀请列表与已进入房间的背景画面。"""
        return (self._read_ocr("在线跨区好友", (0.48, 0.10, 0.82, 0.25), frame, exact=False) is not None
                or self._read_ocr("邀请", (0.45, 0.65, 0.75, 0.90), frame, exact=True) is not None)

    def _resume_invite_host_stage(self, screen, found, frame):
        """邀请中页面意外回退时，按实际识别到的建房步骤继续。"""
        self._invite_phase = None
        self._invite_name_box = None
        self._pending_transition = None
        self._note_screen(screen)
        _, feat, box, score = found
        self.log_warning(f"邀请流程回溯：当前是「{screen}」，重新执行这一步")
        self._host_build(screen, feat, box, score, frame)

    def _recheck_invite_scene(self, frame, name_found=None):
        """点击 2.5 秒无变化时，重新辨认邀请列表、队伍房间或之前的建房页。"""
        self._invite_action_at = time.time()
        found = name_found or self._detect_ocr("等待指定队友", frame)
        if self._invite_popup_visible(frame) or found is not None:
            box = found[2] if found is not None else self._invite_name_box
            if box is not None and self._selected_friend_marker(box):
                self._invite_phase = "invite"
                self._invite_name_box = box
                self.log_info("回溯识别：仍在邀请列表，目标右侧已有对号；继续识别并点击「邀请」")
                return
            if found is not None:
                self._invite_name_box = found[2]
                self._invite_phase = "confirm_selection"
                if self._click(f"{self._invite_target_name()}重试", found[2]):
                    self._invite_action_at = time.time()
                    self.log_warning(f"回溯识别：仍在跨区邀请列表，目标没有对号；已重新点击「{self._invite_target_name()}」")
                return
            cross = self._detect_ocr("跨区确认", frame)
            if cross is not None:
                self._resume_invite_host_stage("跨区确认", cross, frame)
                return
            self.log_warning("回溯识别：邀请弹窗仍在，但暂未找到目标或跨区入口；等待下一张截图")
            return

        if self._room_teammate_visible(frame):
            self._invite_phase = None
            self._host_phase = "waiting_teammate"
            self._room_joined_frames = 0
            self.log_info(f"回溯识别：已进入队伍房间并看到「{self._invite_target_name()}」，改为等待挑战")
            return
        for screen in ("跨区确认", "队伍空位-加号", "创建房间", "求援设置"):
            found = self._detect_host_stage(screen, frame)
            if found is not None:
                self._resume_invite_host_stage(screen, found, frame)
                return
        self.log_warning("回溯识别：当前画面不属于已知邀请或建房步骤，继续等待并重新截图")

    def _invite_step(self, frame):
        """先确认目标好友右侧的选中标注，再点击邀请。"""
        if self._invite_phase == "confirm_selection":
            found = self._detect_ocr("等待指定队友", frame)
            if found is not None:
                current_box = found[2]
                original_box = self._invite_name_box
                # 列表滚动或切页后，不能把其他好友的对号误当作目标已选中。
                if (abs(current_box.x - original_box.x) <= max(25, original_box.height * 2)
                        and abs(current_box.y - original_box.y) <= max(20, original_box.height)):
                    if self._selected_friend_marker(current_box):
                        self._invite_phase = "invite"
                        self._invite_action_at = time.time()
                        self.log_info(f"已识别到「{self._invite_target_name()}」右侧标注「成功点击邀请」，准备点击邀请")
                        return
            now = time.time()
            if now - self._invite_action_at >= INVITE_RECHECK_SECONDS:
                self._recheck_invite_scene(frame, found)
            return

        if self._invite_phase == "invite":
            if self._invite_name_box is not None and not self._selected_friend_marker(self._invite_name_box):
                if time.time() - self._invite_action_at >= INVITE_RECHECK_SECONDS:
                    self._recheck_invite_scene(frame)
                return
            found = self._detect_ocr("邀请队友", frame)
            if found is None:
                now = time.time()
                if now - self._invite_action_at >= INVITE_RECHECK_SECONDS:
                    self._recheck_invite_scene(frame)
                return
            _, _, box, _ = found
            word = (getattr(box, "name", None) or getattr(box, "text", "")).strip()
            if word != "邀请":
                return
            if self._click("邀请", box):
                self._host_invite_sent = True
                self._invite_phase = None
                self._host_phase = "waiting_teammate"
                self._room_joined_frames = 0
                self._phase_clicked_at = time.time()
                self.log_info(f"已选中「{self._invite_target_name()}」并点击「邀请」，等待房间里出现队员姓名")
            else:
                self._invite_action_at = time.time()

    def _expect_next_screen(self, screen):
        """点击发出后只记录期望；下一轮拿新截图确认，不能在本轮盲目前进。"""
        if screen not in HOST_FLOW:
            return
        index = HOST_FLOW.index(screen)
        if screen == "契灵之境入口":
            expected = ("契灵之境页面",)
        elif index + 1 < len(HOST_FLOW):
            expected = (HOST_FLOW[index + 1],)
        else:
            return
        self._pending_transition = (screen, expected, time.time())
        self._pending_last_report = 0.0
        self.log_info(f"等待页面切换：{screen} -> {' / '.join(expected)}")

    def _verify_pending_transition(self, frame, screen):
        """只在下一页被新截图确认后放行；旧页超时才允许重试同一个按钮。"""
        pending = self._pending_transition
        if pending is None:
            return True
        previous, expected, since = pending
        if "契灵之境页面" in expected:
            arrived = self._qiling_page_visible(frame)
        else:
            arrived = screen in expected
        if arrived:
            self._pending_transition = None
            self.log_info(f"页面切换已确认：{previous} -> {' / '.join(expected)}")
            if "契灵之境页面" in expected:
                self.setup_step = HOST_FLOW.index("选择目标-镇墓兽")
            elif screen in HOST_FLOW:
                self.setup_step = HOST_FLOW.index(screen)
            # 下一轮再决定下一步，避免同一张画面连续点击两个按钮。
            return False
        elapsed = time.time() - since
        retry_after = self._retry_interval()
        if previous == "选择目标-镇墓兽":
            retry_after = max(retry_after, TARGET_TO_SUPPORT_RETRY_SECONDS)
        # 跨区后只等待指定队友；没看到名字就不再点弹窗或回退。
        if previous != "跨区确认" and screen == previous and elapsed >= retry_after:
            self._pending_transition = None
            self.log_warning(f"点击「{previous}」后 {elapsed:.1f} 秒仍是原页面，准备重试本步")
            return True
        if elapsed >= self._unknown_report_interval() and time.time() - self._pending_last_report >= 10:
            self._pending_last_report = time.time()
            self.log_warning(f"等待「{previous}」切换至{' / '.join(expected)}，当前识别为{screen or '未知'}；暂不执行后续操作")
            if screen is None:
                # 等不到目标页面时，实时 OCR 全文：直接看画面上有哪些字、目标字在哪
                self._dump_ocr_texts(frame, reason=f"等待切换至「{' / '.join(expected)}」，实时 OCR 全文")
        return False

    def _host_build(self, screen, feat, box, score, frame=None):
        """建房：以**实际识别到的界面**校正进度，再点该界面的按钮。

        这是"不盲推"的关键——点完不立刻前进，下一轮重新识别：
        - 识别到更靠后的界面 -> 说明点生效了，推进；
        - 识别到还是同一个界面 -> 说明没点中，重试（超过上限就告警）；
        - 识别到更靠前的界面 -> 说明被弹回去了，以画面为准回退。
        """
        idx = HOST_FLOW.index(screen)
        prev = self.setup_step
        if idx > prev:
            self.log_info(f"流程推进：第 {prev + 1} 步 -> 第 {idx + 1} 步「{screen}」"
                          f"（识别到 {feat} 分数 {score:.3f}）")
            self.setup_step = idx
        elif idx < prev:
            self.log_warning(f"流程回退：之前以为在第 {prev + 1} 步，但画面显示在第 {idx + 1} 步"
                             f"「{screen}」（{feat} 分数 {score:.3f}）—— 以画面为准，"
                             f"说明上一步点击没生效或被弹回")
            self.setup_step = idx

        # 点过一次之后**不要每轮狂点**：先等「重试间隔秒数」，给弹窗/新界面时间出现。
        # 等待期间每轮仍会跑 OCR 去认后续窗口（例如点完求援后认"创建"两个字）。
        # 这既是用户要的"先点求援，然后等待识别创建"，也能避免一直重复点同一个按钮。
        clicks = getattr(self, "_screen_clicks", 0)
        interval = self._retry_interval()
        if clicks >= MAX_RETRY_ON_SCREEN:
            interval *= 4  # 重试多次仍没变化 -> 大幅放慢，只作兜底
        if clicks >= 1 and time.time() - getattr(self, "_last_click_time", 0.0) < interval:
            return
        if clicks == MAX_RETRY_ON_SCREEN:
            self.log_warning(f"在「{screen}」上已点击 {clicks} 次界面仍未变化："
                             f"点击可能没生效（窗口未激活 / 未以管理员运行 / 坐标偏移），"
                             f"或该按钮标注不准。已放慢重试速度，继续等待画面变化")
        if self._click(feat, box):
            self._screen_clicks = clicks + 1
            self._last_click_time = time.time()
            self._expect_next_screen(screen)

    def _binding_success_visible(self, frame):
        """结契结束回到镇墓兽结契页，且求援再次可用时才开始下一轮。"""
        return (self._read_ocr("镇墓兽", (0, 0, 1, 1), frame, exact=False) is not None
                and self._read_ocr("求援", (0.55, 0.50, 1, 1), frame, exact=True) is not None)

    def _reset_after_success(self):
        self._host_phase = None
        self._invite_phase = None
        self._host_invite_sent = False
        self._pending_transition = None
        self.setup_step = HOST_FLOW.index("求援设置")
        self._room_joined_frames = 0
        self.log_info("已确认结契成功：画面有「镇墓兽」且「求援」可用；从求援开始下一轮")

    def _host_cycle_step(self, frame):
        """首次邀请后反复执行：入房、挑战、准备、结契、继续邀请。"""
        phase = self._host_phase
        self._note_screen(f"队长循环-{phase}")
        if phase in ("battle", "waiting_continue", "waiting_reinvite_dialog") and self._binding_success_visible(frame):
            self._reset_after_success()
            return

        # 失败页可能在任何等待阶段出现；按当前画面处理，避免卡在「等待准备」。
        if phase in ("waiting_teammate", "waiting_auto_teammate", "waiting_ready", "battle"):
            abandon = self._read_ocr("放弃结契", (0, 0, 1, 1), frame, exact=False)
            if abandon is not None and self._click("放弃结契", abandon):
                self._host_phase = "waiting_continue"
                self._phase_clicked_at = time.time()
                self.log_info("已识别结契失败页并点击「放弃结契」，等待续邀提示")
                return

        if phase in ("waiting_teammate", "waiting_auto_teammate"):
            auto_reinvite = phase == "waiting_auto_teammate"
            if not auto_reinvite and time.time() - getattr(self, "_phase_clicked_at", 0) >= INVITE_RECHECK_SECONDS \
                    and self._invite_popup_visible(frame):
                self._host_phase = None
                self._invite_phase = "confirm_selection"
                self.log_warning("点击邀请后 2.5 秒仍停留在邀请列表，回溯识别当前步骤")
                self._recheck_invite_scene(frame)
                return
            teammate_seen = (self._room_teammate_visible(frame, skip_popup_check=True)
                             if auto_reinvite else self._room_teammate_visible(frame))
            if not teammate_seen:
                self._room_joined_frames = 0
                if time.time() - getattr(self, "_last_phase_report", 0) >= self._unknown_report_interval():
                    self._last_phase_report = time.time()
                    self.log_info(f"自动续邀已确认，等待队员直接入房并识别「{self._invite_target_name()}」"
                                  if auto_reinvite else f"邀请已发出，等待在房间中识别到「{self._invite_target_name()}」")
                return
            challenge = self._room_challenge_box(frame)
            if challenge is None:
                self._room_joined_frames = 0
                if time.time() - getattr(self, "_last_phase_report", 0) >= self._unknown_report_interval():
                    self._last_phase_report = time.time()
                    self.log_warning(f"已识别到房间中的「{self._invite_target_name()}」，但尚未匹配到右下角标注「挑战」")
                return
            # 队友姓名和挑战按钮连续两帧同时出现，才确认已进入可开战的房间。
            self._room_joined_frames += 1
            if self._room_joined_frames < 2:
                return
            if self._click("挑战", challenge):
                self._host_phase = "waiting_ready"
                self._phase_clicked_at = time.time()
                self._room_joined_frames = 0
                self.log_info(f"已连续两帧识别到房间中的「{self._invite_target_name()}」和挑战按钮，已点击「挑战」，等待准备按钮")
            return

        if phase == "waiting_ready":
            ready = self._read_ocr("准备", (0.45, 0.55, 1, 1), frame, exact=False)
            if ready is not None and self._click("准备", ready):
                self._host_phase = "battle"
                self._phase_clicked_at = time.time()
                self.log_info("队长已点击「准备」，等待战斗结果")
            elif time.time() - self._phase_clicked_at >= self._retry_interval():
                challenge = self._room_challenge_box(frame) if self._room_teammate_visible(frame) else None
                if challenge is not None and self._click("挑战重试", challenge):
                    self._phase_clicked_at = time.time()
            return

        if phase == "battle":
            if time.time() - self._phase_clicked_at >= self._retry_interval():
                ready = (self._read_ocr("准备", (0.45, 0.55, 1, 1), frame)
                         or self._read_ocr("准备开始", (0.45, 0.55, 1, 1), frame))
                if ready is not None and self._click("准备重试", ready):
                    self._phase_clicked_at = time.time()
                    return
                if self._resume_room_if_ready(frame):
                    return
            continue_box = self._read_ocr("点击屏幕继续", (0.05, 0.60, 0.95, 1), frame, exact=False)
            if continue_box is not None and self._click("点击屏幕继续", continue_box):
                self._host_phase = "waiting_reinvite_dialog"
                self._phase_clicked_at = time.time()
                self.log_info("战斗结果页没有「放弃结契」，已点击继续；等待续邀或求援页面")
            return

        if phase == "waiting_continue":
            # 点击后留 1.5 秒给页面切换；失败页也可能直接显示续邀弹窗。
            if time.time() - self._phase_clicked_at < 1.5:
                return
            label = self._read_ocr("默认邀请队友", (0.10, 0.15, 0.90, 0.85), frame, exact=False)
            if label is not None:
                self._host_phase = "waiting_reinvite_dialog"
                self.log_info("放弃结契后已识别到「默认邀请队友」，进入续邀分支")
                return
            if self._resume_room_if_ready(frame):
                return
            continue_box = self._read_ocr("点击屏幕继续", (0.05, 0.60, 0.95, 1), frame, exact=False)
            if continue_box is not None and self._click("点击屏幕继续", continue_box):
                self._host_phase = "waiting_reinvite_dialog"
                self._phase_clicked_at = time.time()
                self.log_info("已点击下方继续提示，等待续邀弹窗")
            elif time.time() - self._phase_clicked_at >= self._retry_interval():
                abandon = self._read_ocr("放弃结契", (0, 0, 1, 1), frame, exact=False)
                if abandon is not None and self._click("放弃结契重试", abandon):
                    self._phase_clicked_at = time.time()
            return

        if phase == "waiting_reinvite_dialog":
            label = self._read_ocr("默认邀请队友", (0.10, 0.15, 0.90, 0.85), frame, exact=False)
            if label is None:
                if self._resume_room_if_ready(frame):
                    return
                if time.time() - self._phase_clicked_at >= self._retry_interval():
                    continue_box = self._read_ocr("点击屏幕继续", (0.05, 0.60, 0.95, 1), frame, exact=False)
                    if continue_box is not None and self._click("点击屏幕继续重试", continue_box):
                        self._phase_clicked_at = time.time()
                return
            self._default_invite_box = label
            if _checkbox_tick_visible(frame, label):
                self._host_phase = "confirm_reinvite"
                self.log_info("续邀弹窗中「默认邀请队友」已勾选，等待点击确定")
                return
            self._default_invite_before = frame.copy()
            cx, cy = _checkbox_center(label)
            side = max(12, int(label.height * .9))
            checkbox = Box(cx-side//2, cy-side//2, side, side, name="默认邀请队友")
            if self._click("默认邀请队友复选框", checkbox):
                self._default_checked_frames = 0
                self._host_phase = "checking_default_invite"
                self.log_info("已点击「默认邀请队友」左侧复选框，等待对号出现")
            return

        if phase == "checking_default_invite":
            label = self._read_ocr("默认邀请队友", (0.10, 0.15, 0.90, 0.85), frame, exact=False)
            if label is None and self._resume_room_if_ready(frame):
                return
            if label is None or not _checkbox_changed(self._default_invite_before, frame, self._default_invite_box) \
                    or not _checkbox_tick_visible(frame, label):
                self._default_checked_frames = 0
                return
            self._default_checked_frames += 1
            if self._default_checked_frames >= 2:
                self._host_phase = "confirm_reinvite"
                self.log_info("已连续两帧确认「默认邀请队友」左侧出现对号，准备点击确定")
            return

        if phase == "confirm_reinvite":
            if self._read_ocr("默认邀请队友", (0.10, 0.15, 0.90, 0.85), frame, exact=False) is None:
                self._resume_room_if_ready(frame)
                return
            confirm = self._read_ocr("确定", (0.20, 0.35, 0.85, 0.95), frame)
            if confirm is not None and self._click("续邀确定", confirm):
                self._host_phase = "waiting_auto_teammate"
                self._room_joined_frames = 0
                self._phase_clicked_at = time.time()
                self.log_info(f"已确定默认邀请队员，等待队员直接入房；识别「{self._invite_target_name()}」和「挑战」，不再选择跨区好友")
            return

        now = time.time()
        if now - self._last_phase_report >= self._unknown_report_interval():
            self._last_phase_report = now
            self.log_warning(f"未知的队长循环阶段：{phase}")

    # ------------------------------------------------------------------ 队友
    def _teammate_step(self, frame):
        # 邀请覆盖在庭院/房间上，先认弹窗，再认背景页面。
        auto_box = self.find_one("队友-选择自动接受", HV, VV, threshold=self._threshold())
        accept_box = self.find_one("队友-选择接受", HV, VV, threshold=self._threshold())
        if auto_box is not None:
            if not self.auto_accept_enabled or time.time() - self._last_auto_accept_click >= 5:
                if self._click("队友-选择自动接受", auto_box):
                    self.auto_accept_enabled = True
                    self._last_auto_accept_click = time.time()
                    self.log_info("队友：已选择自动接受邀请，等待入房")
            return
        if accept_box is not None:
            if self._click("队友-选择接受", accept_box):
                self.log_info("队友：已接受邀请")
            return

        continue_box = self._read_ocr("点击屏幕继续", (0.05, 0.60, 0.95, 1), frame, exact=False)
        if continue_box is not None:
            if time.time() - self._last_continue_click >= self._retry_interval():
                if self._click("点击屏幕继续", continue_box):
                    self._last_continue_click = time.time()
                    self.log_info("队友：已点击下方继续提示")
            return

        ready = (self._read_ocr("准备", (0.45, 0.55, 1, 1), frame)
                 or self._read_ocr("准备开始", (0.45, 0.55, 1, 1), frame))
        if ready is not None:
            if time.time() - self._last_teammate_ready_click >= self._retry_interval():
                if self._click("准备", ready):
                    self._last_teammate_ready_click = time.time()
                    self.log_info("队友：已点击准备，等待战斗")
            return

        screen, feat, box, score = self._detect_screen(frame)
        if screen is None:
            self._on_unknown_screen(frame)
            return
        stayed = self._note_screen(screen)

        if screen == "邀请弹窗-队友":
            return
        if screen == "过场-点击继续":
            self._click(feat, box)
            return
        if screen == "组队房间-队友":
            self._click(feat, box)
            self.log_info("队友：已点准备开始")
            return
        if screen == "放弃结契":
            return  # 只由队长操作
        if screen == "组队房间-队长":
            # 队友窗口却看到队长的挑战按钮（不该发生），不动，避免误点。
            if stayed > 15:
                self.log_warning("队员窗口出现「8-点击挑战」，角色可能选反了（本窗口应为队长）")
            return
        if stayed > 15:
            self.log_info(f"队友当前在「{screen}」，无需操作")

    # ------------------------------------------------------------------ 工具
    def _click(self, name, box):
        """点击已经找到的 box（不重新匹配，避免重复计算）。"""
        sent = self.click_box(box, after_sleep=1.5)
        if not sent:
            self.log_warning(f"点击 {name} 失败")
            return False
        self.log_info(f"已点击「{name}」@ x={getattr(box, 'x', '?')},y={getattr(box, 'y', '?')}")
        return True

    def _try_click(self, name, is_backup=False):
        """找到指定特征就点；点到返回 True，没找到/点失败返回 False。"""
        box = self.find_one(name, HV, VV, threshold=self._threshold(is_backup))
        if not box:
            return False
        return self._click(name, box)
