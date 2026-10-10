"""Checks page recognition and transitions in the QiLing workflow."""

import unittest
import time
import re
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.tasks.QiLingTask import (BINDING_PAGE_OCR, HOME_VERTICAL_OCR_REGION, HOST_FLOW,
                                   QiLingTask, SCREEN_OCR, _invite_name_pattern,
                                    INVITE_RECHECK_SECONDS, PLUS_OCR_REGION, DEFAULT_INVITE_NAME, ROOM_TEAMMATE_REGION,
                                  TARGET_TO_SUPPORT_RETRY_SECONDS,
                                  _home_vertical_reference, _match_reference,
                                  _marker_matches_friend, _checkbox_tick_visible,
                                  _checkbox_changed, _role_by_window_area)


ROOT = Path(__file__).resolve().parents[1] / "assets" / "qiling_reference"


def screenshot(name):
    path = ROOT / name
    if not path.is_file():
        path = ROOT.parent / "images" / "qiling" / name
    return cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)


class QiLingFlowTest(unittest.TestCase):
    def test_window_area_assigns_roles_from_selected_game_window(self):
        windows = [(101, 0, 0, 1600, 900), (202, 30, 20, 1280, 720)]
        self.assertEqual(_role_by_window_area(101, windows), "队长")
        self.assertEqual(_role_by_window_area(202, windows), "队员")
        self.assertIsNone(_role_by_window_area(303, windows))

    def test_window_area_tie_uses_top_left_window(self):
        windows = [(101, 100, 50, 1280, 720), (202, 20, 10, 1280, 720)]
        self.assertEqual(_role_by_window_area(202, windows), "队长")
        self.assertEqual(_role_by_window_area(101, windows), "队员")

    @patch("ok.util.window.get_window_bounds")
    @patch("ok.util.window.find_all_visible_windows")
    def test_auto_role_uses_selected_hwnd_and_can_be_disabled(self, find_windows, bounds):
        find_windows.return_value = [
            (101, "阴阳师", "onmyoji.exe", ""),
            (202, "阴阳师", "onmyoji_future.exe", ""),
            (303, "其他", "notepad.exe", ""),
        ]
        bounds.side_effect = lambda hwnd: (
            (0, 0, 1600, 900, 1600, 900, 1) if hwnd == 101 else
            (20, 20, 1280, 720, 1280, 720, 1)
        )
        task = SimpleNamespace(
            config={"角色": "队长", "按窗口面积自动分配角色": True},
            executor=SimpleNamespace(device_manager=SimpleNamespace(config={"selected_hwnd": 202})),
            log_info=MagicMock(),
        )
        QiLingTask._assign_role_by_window_area(task)
        self.assertEqual(task.config["角色"], "队员")
        self.assertEqual(bounds.call_count, 2)

        task.config["按窗口面积自动分配角色"] = False
        task.config["角色"] = "队长"
        QiLingTask._assign_role_by_window_area(task)
        self.assertEqual(task.config["角色"], "队长")

    def test_dedicated_qiling_feature_set_finds_challenge(self):
        task = QiLingTask(MagicMock(), MagicMock())
        box = task.find_one("挑战", 1.0, 1.0, threshold=.8, frame=screenshot("1.png"))
        self.assertIsNotNone(box)
        self.assertEqual(box.name, "挑战")

    def test_navigation_uses_ocr_in_the_button_region(self):
        calls = []
        box = SimpleNamespace(text="探索", x=600, y=100)
        task = SimpleNamespace(
            ocr=lambda *args, **kwargs: calls.append((args, kwargs)) or [box],
            log_info=lambda *_: None,
            log_warning=lambda *_: None,
        )
        result = QiLingTask._detect_ocr(task, "主界面-探索", screenshot("庭院.png"))
        self.assertEqual(result[0], "主界面-探索")
        self.assertEqual(calls[0][0], SCREEN_OCR["主界面-探索"][1])
        self.assertEqual(calls[0][1]["match"], "探索")

        calls.clear()
        box = SimpleNamespace(name="妖气探索契灵之境", x=210, y=600,
                              width=510, height=22, confidence=.98)
        result = QiLingTask._detect_ocr(task, "契灵之境入口", screenshot("探索.png"))
        self.assertEqual(calls[0][0], SCREEN_OCR["契灵之境入口"][1])
        self.assertIsNotNone(calls[0][1]["match"].search("妖气探索契灵之境"))
        self.assertEqual(result[0], "契灵之境入口")
        self.assertEqual(result[2].name, "契灵之境")
        self.assertEqual(result[2].x, 720 - 22 * 4)
        self.assertEqual(result[2].width, 22 * 4)

        calls.clear()
        QiLingTask._detect_ocr(task, "求援设置", screenshot("契灵之境.png"))
        self.assertEqual(calls[0][0], SCREEN_OCR["求援设置"][1])
        self.assertEqual(calls[0][1]["match"], "求援")

    def test_qiling_entrance_ignores_matching_text_above_bottom_bar(self):
        upper = SimpleNamespace(name="契灵之境", x=650, y=420,
                                width=90, height=22, confidence=.99)
        bottom = SimpleNamespace(name="妖气探索契灵之境", x=210, y=600,
                                 width=510, height=22, confidence=.98)
        task = SimpleNamespace(
            ocr=lambda *args, **kwargs: [upper, bottom],
            log_info=lambda *_: None,
            log_warning=lambda *_: None,
        )
        found = QiLingTask._detect_ocr(task, "契灵之境入口", screenshot("探索.png"))
        self.assertEqual(found[2].x, 720 - 22 * 4)

    def test_named_qiling_screenshot_confirms_only_qiling_page(self):
        task = SimpleNamespace(_detect_ocr=lambda *_: None)
        self.assertTrue(QiLingTask._qiling_page_visible(task, screenshot("契灵之境.png")))
        self.assertTrue(QiLingTask._qiling_page_visible(task, screenshot("4.png")))
        self.assertFalse(QiLingTask._qiling_page_visible(task, screenshot("庭院.png")))
        self.assertFalse(QiLingTask._qiling_page_visible(task, screenshot("探索.png")))

    def test_second_courtyard_joins_vertical_ocr_characters(self):
        calls = []
        glyphs = [
            SimpleNamespace(text="探", x=380, y=200, width=25, height=28, confidence=.98),
            SimpleNamespace(text="索", x=382, y=227, width=22, height=27, confidence=.62),
        ]

        def ocr(*args, **kwargs):
            calls.append((args, kwargs))
            return [] if kwargs.get("match") else glyphs

        task = SimpleNamespace(ocr=ocr, log_warning=lambda *_: None)
        task._detect_ocr = lambda screen, frame: QiLingTask._detect_ocr(task, screen, frame)
        found = QiLingTask._detect_home_explore(task, screenshot("庭院-1.png"))
        self.assertEqual(found[0], "主界面-探索")
        self.assertEqual(found[2].name, "探索")
        self.assertEqual(calls[-1][0], HOME_VERTICAL_OCR_REGION)
        self.assertTrue(380 <= found[2].x <= 405)

    def test_second_courtyard_screenshot_is_a_safe_fallback(self):
        reference = _home_vertical_reference()
        search = (.25, .20, .50, .52)
        self.assertIsNotNone(_match_reference(screenshot("庭院-1.png"), reference, search, .78))
        for other in ("庭院.png", "探索.png", "契灵之境.png"):
            self.assertIsNone(_match_reference(screenshot(other), reference, search, .78))

    def test_click_waits_for_next_page_before_advancing(self):
        messages = []
        task = SimpleNamespace(
            _pending_transition=None,
            _pending_last_report=0.0,
            setup_step=0,
            log_info=messages.append,
            log_warning=messages.append,
            _retry_interval=lambda: 4.0,
            _unknown_report_interval=lambda: 10.0,
            _qiling_page_visible=lambda _: False,
        )
        QiLingTask._expect_next_screen(task, HOST_FLOW[0])
        self.assertFalse(QiLingTask._verify_pending_transition(task, None, HOST_FLOW[0]))
        self.assertIsNotNone(task._pending_transition)
        self.assertFalse(QiLingTask._verify_pending_transition(task, None, HOST_FLOW[1]))
        self.assertIsNone(task._pending_transition)

        QiLingTask._expect_next_screen(task, HOST_FLOW[1])
        self.assertFalse(QiLingTask._verify_pending_transition(task, None, HOST_FLOW[1]))
        task._qiling_page_visible = lambda _: True
        self.assertFalse(QiLingTask._verify_pending_transition(task, None, "契灵之境页面"))
        self.assertIsNone(task._pending_transition)
        self.assertEqual(task.setup_step, HOST_FLOW.index("选择目标-镇墓兽"))

    def test_target_click_waits_for_support_ocr_before_next_click(self):
        messages = []
        task = SimpleNamespace(
            config={"OCR识别界面": "创建房间"},
            role="host",
            setup_step=HOST_FLOW.index("选择目标-镇墓兽"),
            _last_screen="选择目标-镇墓兽",
            _pending_transition=None,
            _pending_last_report=0.0,
            log_info=messages.append,
            log_warning=messages.append,
            _retry_interval=lambda: 4.0,
            _unknown_report_interval=lambda: 10.0,
        )
        QiLingTask._expect_next_screen(task, "选择目标-镇墓兽")
        screens = QiLingTask._ocr_screens(task)
        self.assertIn("求援设置", screens)
        self.assertNotIn("创建房间", screens)

        previous, expected, _ = task._pending_transition
        task._pending_transition = (previous, expected, time.time() - 5)
        self.assertFalse(QiLingTask._verify_pending_transition(task, None, previous))
        self.assertIsNotNone(task._pending_transition)

        self.assertFalse(QiLingTask._verify_pending_transition(task, None, "求援设置"))
        self.assertIsNone(task._pending_transition)
        self.assertEqual(task.setup_step, HOST_FLOW.index("求援设置"))
        screens = QiLingTask._ocr_screens(task)
        self.assertLess(screens.index("求援设置"), screens.index("创建房间"))

        task._pending_transition = (previous, expected, time.time() - TARGET_TO_SUPPORT_RETRY_SECONDS - 1)
        self.assertTrue(QiLingTask._verify_pending_transition(task, None, previous))
        self.assertIsNone(task._pending_transition)

    def test_after_support_flow_confirms_each_new_screen(self):
        task = SimpleNamespace(
            _pending_transition=None,
            _pending_last_report=0.0,
            _host_invite_sent=False,
            setup_step=HOST_FLOW.index("求援设置"),
            log_info=lambda *_: None,
            log_warning=lambda *_: None,
            _retry_interval=lambda: 4.0,
            _unknown_report_interval=lambda: 10.0,
        )
        expected_order = ["求援设置", "创建房间", "队伍空位-加号", "跨区确认",
                          "等待指定队友"]
        self.assertEqual(HOST_FLOW[3:], expected_order)
        for previous, next_screen in zip(expected_order, expected_order[1:]):
            QiLingTask._expect_next_screen(task, previous)
            self.assertEqual(task._pending_transition[1], (next_screen,))
            self.assertFalse(QiLingTask._verify_pending_transition(task, None, next_screen))
            self.assertEqual(task.setup_step, HOST_FLOW.index(next_screen))
            self.assertIsNone(task._pending_transition)
        self.assertFalse(task._host_invite_sent)

    def test_cross_region_waits_indefinitely_for_target_name(self):
        task = SimpleNamespace(
            _pending_transition=None,
            _pending_last_report=0.0,
            _host_invite_sent=False,
            setup_step=HOST_FLOW.index("跨区确认"),
            log_info=lambda *_: None,
            log_warning=lambda *_: None,
            _retry_interval=lambda: 4.0,
            _unknown_report_interval=lambda: 10.0,
        )
        QiLingTask._expect_next_screen(task, "跨区确认")
        previous, expected, _ = task._pending_transition
        task._pending_transition = (previous, expected, time.time() - 120)
        self.assertFalse(QiLingTask._verify_pending_transition(task, None, previous))
        self.assertIsNotNone(task._pending_transition)
        self.assertFalse(task._host_invite_sent)

    def test_plus_ocr_uses_middle_and_lower_region(self):
        calls = []
        box = SimpleNamespace(text="+", confidence=.91)
        task = SimpleNamespace(
            ocr=lambda *args, **kwargs: calls.append((args, kwargs)) or [box],
            log_warning=lambda *_: None,
        )
        got = QiLingTask._detect_plus(task, screenshot("19.png"))
        self.assertEqual(got[0], "队伍空位-加号")
        self.assertEqual(calls[0][0], PLUS_OCR_REGION)
        self.assertEqual(calls[0][1]["match"], "+")
        self.assertEqual(calls[0][1]["frame"].shape[2], 3)
        self.assertTrue(np.array_equal(calls[0][1]["frame"][:, :, 0], calls[0][1]["frame"][:, :, 1]))

        calls.clear()
        task.ocr = lambda *args, **kwargs: calls.append(kwargs) or ([] if len(calls) == 1 else [box])
        self.assertEqual(QiLingTask._detect_plus(task, screenshot("19.png"))[0], "队伍空位-加号")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["frame"].shape[2], 3)

    def test_new_popup_takes_priority_over_old_support_button(self):
        task = SimpleNamespace(
            _pending_transition=("求援设置", ("创建房间",), time.time()),
            setup_step=HOST_FLOW.index("求援设置"),
            _detect_host_stage=lambda screen, _: (screen, "OCR", object(), 1.0),
        )
        found = QiLingTask._detect_host_screen(task, None)
        self.assertEqual(found[0], "创建房间")

    def test_target_name_accepts_common_ocr_lookalike(self):
        _, region = SCREEN_OCR["等待指定队友"]
        pattern = _invite_name_pattern(DEFAULT_INVITE_NAME)
        self.assertIsNotNone(pattern.search("59 过往的兔耳2"))
        self.assertIsNotNone(pattern.search("过往的免耳２"))
        self.assertIsNone(pattern.search("过往的兔耳"))
        self.assertIsNone(pattern.search("过往的兔耳20"))
        self.assertIsNone(pattern.search("过往的兔子"))
        custom = _invite_name_pattern("晴明.一号")
        self.assertIsNotNone(custom.search("59 晴明.一号"))
        self.assertIsNone(custom.search("晴明X一号"))
        self.assertLess(region[1], .35)

    def test_configured_invite_name_is_used_for_ocr(self):
        calls = []
        box = SimpleNamespace(name="月见", x=100, y=100, width=40, height=18)
        task = SimpleNamespace(
            _invite_target_name=lambda: "月见",
            ocr=lambda *args, **kwargs: calls.append((args, kwargs)) or [box],
            log_info=lambda *_: None,
            log_warning=lambda *_: None,
        )
        task._read_ocr = lambda *args, **kwargs: QiLingTask._read_ocr(task, *args, **kwargs)
        found = QiLingTask._detect_ocr(task, "等待指定队友", None)
        self.assertEqual(found[2], box)
        self.assertIsNone(calls[0][1]["match"])

    def test_invite_ocr_accepts_traditional_variant_without_matching_other_names(self):
        frame = np.zeros((751, 1331, 3), dtype=np.uint8)
        target = SimpleNamespace(name="天青色的煙雨", x=350, y=220, width=150, height=25)
        other = SimpleNamespace(name="天青色的细雨", x=350, y=280, width=150, height=25)
        boxes = [other, target]
        task = SimpleNamespace(
            _invite_target_name=lambda: "天青色的烟雨",
            ocr=lambda *_args, **_kwargs: boxes,
            log_info=lambda *_: None,
            log_warning=lambda *_: None,
        )
        task._read_ocr = lambda *args, **kwargs: QiLingTask._read_ocr(task, *args, **kwargs)
        found = QiLingTask._detect_ocr(task, "等待指定队友", frame)
        self.assertIs(found[2], target)
        task._invite_popup_visible = lambda _: False
        self.assertTrue(QiLingTask._room_teammate_visible(task, frame))
        boxes.remove(target)
        self.assertIsNone(QiLingTask._detect_ocr(task, "等待指定队友", frame))
        self.assertFalse(QiLingTask._room_teammate_visible(task, frame))

    def test_chinese_role_selection_maps_to_internal_behavior(self):
        task = SimpleNamespace(config={"角色": "队员"}, instance_index=1, role=None,
                               log_info=lambda *_: None)
        self.assertEqual(QiLingTask._resolve_role(task), "teammate")
        task.config["角色"] = "队长"
        self.assertEqual(QiLingTask._resolve_role(task), "host")

    def test_target_is_clicked_then_invited_only_after_annotated_checkmark(self):
        name = "等待指定队友"
        name_box = SimpleNamespace(name="过往的兔耳2", x=110, y=90, width=65, height=16)
        invite_box = SimpleNamespace(name="邀请", x=220, y=155, width=35, height=18)
        marker = SimpleNamespace(x=186, y=88, width=16, height=18)
        other_row_marker = SimpleNamespace(x=186, y=136, width=16, height=18)
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        clicks = []
        logs = []
        marker_state = [None]
        task = SimpleNamespace(
            _invite_phase=None,
            _host_phase=None,
            _invite_name_box=None,
            _invite_target_name=lambda: "过往的兔耳2",
            _invite_last_report=0.0,
            _host_invite_sent=False,
            _read_ocr=lambda *_args, **_kwargs: None,
            _host_scene_action=lambda _: False,
            _resume_room_if_ready=lambda _: False,
            _promote_binding_page=lambda _: None,
            _detect_host_screen=lambda _: (name, "OCR:过往的兔耳2", name_box, 1.0),
            _verify_pending_transition=lambda *_: True,
            _note_screen=lambda _: 0,
            _click=lambda label, _: clicks.append(label) or True,
            _detect_ocr=lambda stage, _: (stage, "OCR", name_box if stage == name else invite_box, 1.0),
            _selected_friend_marker=lambda box: (marker_state[0] if _marker_matches_friend(marker_state[0], box) else None),
            _unknown_report_interval=lambda: 10.0,
            log_info=logs.append,
            log_warning=logs.append,
        )
        QiLingTask._host_step(task, frame)
        self.assertEqual(clicks, ["过往的兔耳2"])
        self.assertFalse(task._host_invite_sent)

        QiLingTask._invite_step(task, frame)
        self.assertEqual(task._invite_phase, "confirm_selection")
        self.assertEqual(clicks, ["过往的兔耳2"])
        marker_state[0] = other_row_marker
        QiLingTask._invite_step(task, frame)
        self.assertEqual(task._invite_phase, "confirm_selection")

        marker_state[0] = marker
        QiLingTask._invite_step(task, frame)
        self.assertEqual(task._invite_phase, "invite")
        QiLingTask._invite_step(task, frame)
        self.assertEqual(clicks, ["过往的兔耳2", "邀请"])
        self.assertTrue(task._host_invite_sent)

    def test_already_selected_friend_skips_second_name_click(self):
        name = "等待指定队友"
        name_box = SimpleNamespace(x=458, y=204, width=134, height=30)
        marker = SimpleNamespace(x=624, y=189, width=38, height=36)
        self.assertTrue(_marker_matches_friend(marker, name_box))
        self.assertFalse(_marker_matches_friend(SimpleNamespace(x=624, y=300, width=38, height=36), name_box))
        clicks = []
        task = SimpleNamespace(
            _host_phase=None,
            _invite_phase=None,
            _invite_name_box=None,
            _invite_target_name=lambda: "过往的兔耳2",
            _read_ocr=lambda *_args, **_kwargs: None,
            _host_scene_action=lambda _: False,
            _resume_room_if_ready=lambda _: False,
            _promote_binding_page=lambda _: None,
            _detect_host_screen=lambda _: (name, "OCR:过往的兔耳2", name_box, 1.0),
            _verify_pending_transition=lambda *_: True,
            _note_screen=lambda _: 0,
            _selected_friend_marker=lambda _: marker,
            _click=lambda label, _: clicks.append(label) or True,
            log_info=lambda *_: None,
        )
        QiLingTask._host_step(task, screenshot("0.png"))
        self.assertEqual(task._invite_phase, "invite")
        self.assertEqual(clicks, [])

    def test_missing_checkmark_rechecks_scene_and_retries_after_2_5_seconds(self):
        name = "等待指定队友"
        name_box = SimpleNamespace(x=458, y=204, width=134, height=30)
        frame = screenshot("0.png")
        clicks = []
        task = SimpleNamespace(
            _invite_phase="confirm_selection",
            _invite_name_box=name_box,
            _invite_target_name=lambda: "过往的兔耳2",
            _invite_action_at=time.time() - INVITE_RECHECK_SECONDS - .1,
            _detect_ocr=lambda stage, _: (stage, "OCR", name_box, 1.0) if stage == name else None,
            _selected_friend_marker=lambda _: None,
            _invite_popup_visible=lambda _: True,
            _click=lambda label, _: clicks.append(label) or True,
            log_warning=lambda *_: None,
        )
        task._recheck_invite_scene = lambda image, found: QiLingTask._recheck_invite_scene(task, image, found)
        QiLingTask._invite_step(task, frame)
        self.assertEqual(clicks, ["过往的兔耳2重试"])
        self.assertEqual(task._invite_phase, "confirm_selection")
        QiLingTask._invite_step(task, frame)
        self.assertEqual(clicks, ["过往的兔耳2重试"])

    def test_invite_recheck_dispatches_the_visible_scene(self):
        frame = screenshot("0.png")
        seen = []
        cross = ("跨区确认", "OCR:跨区", object(), 1.0)
        task = SimpleNamespace(
            _invite_phase="confirm_selection",
            _invite_name_box=None,
            _invite_target_name=lambda: "过往的兔耳2",
            _invite_popup_visible=lambda _: True,
            _detect_ocr=lambda stage, _: cross if stage == "跨区确认" else None,
            _resume_invite_host_stage=lambda stage, found, _: seen.append((stage, found)),
            log_warning=lambda *_: None,
        )
        QiLingTask._recheck_invite_scene(task, frame)
        self.assertEqual(seen, [("跨区确认", cross)])

        task._invite_popup_visible = lambda _: False
        task._room_teammate_visible = lambda _: True
        task._room_joined_frames = 1
        task.log_info = lambda *_: None
        QiLingTask._recheck_invite_scene(task, frame)
        self.assertIsNone(task._invite_phase)
        self.assertEqual(task._host_phase, "waiting_teammate")
        self.assertEqual(task._room_joined_frames, 0)

    def test_invite_recheck_returns_to_plus_step_if_popup_closed_without_room(self):
        frame = screenshot("1.png")
        plus = ("队伍空位-加号", "OCR:+", object(), 1.0)
        seen = []
        task = SimpleNamespace(
            _invite_popup_visible=lambda _: False,
            _detect_ocr=lambda *_: None,
            _room_teammate_visible=lambda _: False,
            _detect_host_stage=lambda stage, _: plus if stage == "队伍空位-加号" else None,
            _resume_invite_host_stage=lambda stage, found, _: seen.append((stage, found)),
            log_warning=lambda *_: None,
        )
        QiLingTask._recheck_invite_scene(task, frame)
        self.assertEqual(seen, [("队伍空位-加号", plus)])

    def test_invite_click_still_on_popup_returns_to_selection_check(self):
        frame = screenshot("0.png")
        state = []
        task = SimpleNamespace(
            _host_phase="waiting_teammate",
            _invite_target_name=lambda: "过往的兔耳2",
            _invite_phase=None,
            _phase_clicked_at=time.time() - INVITE_RECHECK_SECONDS - .1,
            _note_screen=lambda *_: None,
            _read_ocr=lambda *_args, **_kwargs: None,
            _invite_popup_visible=lambda _: True,
            _recheck_invite_scene=lambda _: state.append((task._host_phase, task._invite_phase)),
            log_warning=lambda *_: None,
        )
        QiLingTask._host_cycle_step(task, frame)
        self.assertEqual(state, [(None, "confirm_selection")])

    def test_room_waits_for_teammate_name_before_challenge(self):
        clicks = []
        teammate_states = iter((False, True, True))
        challenge = SimpleNamespace(name="挑战")
        task = SimpleNamespace(
            _host_phase="waiting_teammate",
            _invite_target_name=lambda: "过往的兔耳2",
            _phase_clicked_at=time.time(),
            _room_joined_frames=0,
            _invite_popup_visible=lambda _: False,
            _room_teammate_visible=lambda _: next(teammate_states),
            _room_challenge_box=lambda _: challenge,
            _note_screen=lambda _: None,
            _read_ocr=lambda *_args, **_kwargs: None,
            _unknown_report_interval=lambda: 10.0,
            _click=lambda label, _: clicks.append(label) or True,
            log_info=lambda *_: None,
        )
        for _ in range(2):
            QiLingTask._host_cycle_step(task, None)
            self.assertEqual(clicks, [])
        QiLingTask._host_cycle_step(task, None)
        self.assertEqual(clicks, ["挑战"])
        self.assertEqual(task._host_phase, "waiting_ready")

    def test_room_teammate_uses_name_ocr_and_challenge_annotation(self):
        frame = np.zeros((751, 1331, 3), dtype=np.uint8)
        challenge = SimpleNamespace(x=1215, y=620, width=105, height=96, name="挑战")
        room_label = SimpleNamespace(name="过往的兔耳2")
        calls = []
        task = SimpleNamespace(
            _read_ocr=lambda word, region, *_args, **_kwargs: calls.append((word, region)) or
                      (room_label if word == "过往的兔耳2" else None),
            _invite_target_name=lambda: "过往的兔耳2",
            _invite_popup_visible=lambda _: False,
            find_one=lambda name, *_args, **kwargs: calls.append((name, kwargs)) or challenge,
            _threshold=lambda: .8,
        )
        self.assertTrue(QiLingTask._room_teammate_visible(task, frame))
        self.assertEqual(calls[:1], [("过往的兔耳2", ROOM_TEAMMATE_REGION)])
        self.assertIs(QiLingTask._room_challenge_box(task, frame), challenge)
        self.assertEqual(calls[-1][0], "挑战")
        self.assertIs(calls[-1][1]["frame"], frame)

        task.find_one = lambda *_args, **_kwargs: SimpleNamespace(x=50, y=40, width=105, height=96)
        self.assertIsNone(QiLingTask._room_challenge_box(task, frame))

        annotations = json.loads((ROOT.parent / "qiling_annotations.json").read_text(encoding="utf-8"))
        self.assertIn("挑战", {category["name"] for category in annotations["categories"]})

    def test_room_teammate_ignores_friend_invitation_popup(self):
        frame = screenshot("0.png")
        seen = []
        task = SimpleNamespace(
            _invite_popup_visible=lambda _: True,
            _read_ocr=lambda word, *_args, **_kwargs: seen.append(word) or SimpleNamespace(name=word),
            _invite_target_name=lambda: "过往的兔耳2",
        )
        self.assertFalse(QiLingTask._room_teammate_visible(task, frame))
        self.assertEqual(seen, [])
        self.assertTrue(QiLingTask._room_teammate_visible(task, frame, skip_popup_check=True))
        self.assertEqual(seen, ["过往的兔耳2"])

    def test_room_footer_only_invite_does_not_look_like_invitation_popup(self):
        boxes = [SimpleNamespace(name="权限不公开（仅邀请）")]
        task = SimpleNamespace(
            ocr=lambda *_args, **_kwargs: boxes,
            log_warning=lambda *_: None,
        )
        task._read_ocr = lambda *args, **kwargs: QiLingTask._read_ocr(task, *args, **kwargs)
        self.assertFalse(QiLingTask._invite_popup_visible(task, None))
        boxes[:] = [SimpleNamespace(name="邀请")]
        self.assertTrue(QiLingTask._invite_popup_visible(task, None))

    def test_auto_reinvite_waits_for_teammate_then_clicks_challenge_without_popup_recheck(self):
        clicks = []
        teammate_present = [False]
        challenge = SimpleNamespace(name="挑战")
        task = SimpleNamespace(
            _host_phase="waiting_auto_teammate",
            _invite_target_name=lambda: "过往的兔耳2",
            _phase_clicked_at=time.time() - 10,
            _room_joined_frames=0,
            _read_ocr=lambda *_args, **_kwargs: None,
            _invite_popup_visible=lambda _: self.fail("自动续邀不应回到跨区选人弹窗"),
            _room_teammate_visible=lambda _, skip_popup_check=False: teammate_present[0] and skip_popup_check,
            _room_challenge_box=lambda _: challenge,
            _note_screen=lambda _: None,
            _unknown_report_interval=lambda: 10.0,
            _click=lambda label, _: clicks.append(label) or True,
            log_info=lambda *_: None,
        )
        QiLingTask._host_cycle_step(task, None)
        self.assertEqual(clicks, [])
        teammate_present[0] = True
        QiLingTask._host_cycle_step(task, None)
        QiLingTask._host_cycle_step(task, None)
        self.assertEqual(clicks, ["挑战"])
        self.assertEqual(task._host_phase, "waiting_ready")

    def test_second_auto_cycle_recovers_when_invite_dialog_is_skipped(self):
        clicks = []
        task = SimpleNamespace(
            _host_phase="waiting_reinvite_dialog",
            _phase_clicked_at=time.time() - 10,
            _invite_target_name=lambda: "过往的兔耳2",
            _room_joined_frames=0,
            _binding_success_visible=lambda _: False,
            _read_ocr=lambda *_args, **_kwargs: None,
            _room_teammate_visible=lambda _, skip_popup_check=False: skip_popup_check,
            _room_challenge_box=lambda _: SimpleNamespace(name="挑战"),
            _note_screen=lambda _: None,
            _click=lambda label, _: clicks.append(label) or True,
            log_info=lambda *_: None,
        )
        task._resume_room_if_ready = lambda frame: QiLingTask._resume_room_if_ready(task, frame)
        QiLingTask._host_cycle_step(task, None)
        self.assertEqual(task._host_phase, "waiting_auto_teammate")
        self.assertEqual(task._room_joined_frames, 1)
        QiLingTask._host_cycle_step(task, None)
        self.assertEqual(clicks, ["挑战"])
        self.assertEqual(task._host_phase, "waiting_ready")

    def test_scene_dispatch_keeps_challenging_across_multiple_cycles(self):
        scene = ["room"]
        clicks = []
        button = SimpleNamespace(name="按钮")
        task = SimpleNamespace(
            _host_phase=None,
            _invite_phase=None,
            _pending_transition=None,
            _invite_target_name=lambda: "过往的兔耳2",
            _host_invite_sent=True,
            setup_step=HOST_FLOW.index("等待指定队友"),
            _room_joined_frames=0,
            _phase_clicked_at=0.0,
            _binding_success_visible=lambda _: scene[0] == "success",
            _read_ocr=lambda word, *_args, **_kwargs: button if (
                (scene[0] == "failure" and word == "放弃结契")
                or (scene[0] == "continue" and word == "点击屏幕继续")
                or (scene[0] == "ready" and word == "准备")) else None,
            _room_challenge_box=lambda _: button if scene[0] == "room" else None,
            _room_teammate_visible=lambda _, skip_popup_check=False: scene[0] == "room",
            _retry_interval=lambda: 4.0,
            _note_screen=lambda _: None,
            _click=lambda label, _: clicks.append(label) or True,
            log_info=lambda *_: None,
        )
        task._host_cycle_step = lambda frame: QiLingTask._host_cycle_step(task, frame)
        task._reset_after_success = lambda: QiLingTask._reset_after_success(task)
        for _ in range(3):
            scene[0] = "room"
            self.assertTrue(QiLingTask._host_scene_action(task, None))
            self.assertTrue(QiLingTask._host_scene_action(task, None))
            self.assertEqual(task._host_phase, "waiting_ready")
            scene[0] = "ready"
            QiLingTask._host_scene_action(task, None)
            self.assertEqual(task._host_phase, "battle")
            scene[0] = "failure"
            QiLingTask._host_scene_action(task, None)
            self.assertEqual(task._host_phase, "waiting_continue")
            task._phase_clicked_at -= 2
            scene[0] = "continue"
            QiLingTask._host_scene_action(task, None)
            self.assertEqual(task._host_phase, "waiting_reinvite_dialog")
        self.assertEqual(clicks.count("挑战"), 3)
        scene[0] = "success"
        QiLingTask._host_scene_action(task, None)
        self.assertIsNone(task._host_phase)
        self.assertEqual(task.setup_step, HOST_FLOW.index("求援设置"))

    def test_ocr_grayscale_retry_keeps_three_channels(self):
        frame = screenshot("11.png")
        frames = []
        box = SimpleNamespace(name="挑战")
        task = SimpleNamespace(
            ocr=lambda *_args, **kwargs: frames.append(kwargs["frame"]) or ([] if len(frames) == 1 else [box]),
            log_warning=lambda *_: None,
        )
        self.assertIs(QiLingTask._read_ocr(task, "挑战", (0.68, 0.55, 1, 1), frame), box)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[1].shape[2], 3)
        self.assertTrue(np.array_equal(frames[1][:, :, 0], frames[1][:, :, 1]))

    def test_room_name_ocr_accepts_traditional_variant_with_prefix(self):
        frame = screenshot("0.png")
        label = SimpleNamespace(name="网易-天青色的煙雨")
        matches = []
        def ocr(*_args, **kwargs):
            matches.append(kwargs["match"])
            return [label]
        task = SimpleNamespace(ocr=ocr, log_warning=lambda *_: None)
        self.assertIs(QiLingTask._read_ocr(task, "天青色的烟雨", ROOM_TEAMMATE_REGION,
                                           frame, exact=False, normalize_traditional=True), label)
        self.assertIsNone(matches[0])
        self.assertEqual(len(matches), 1)

    def test_host_waiting_ready_handles_failure_before_ready(self):
        buttons = {word: SimpleNamespace(name=word) for word in ("准备", "放弃结契", "点击屏幕继续")}
        clicks = []
        task = SimpleNamespace(
            _host_phase="waiting_ready",
            _phase_clicked_at=time.time(),
            _retry_interval=lambda: 4.0,
            _binding_success_visible=lambda _: False,
            _read_ocr=lambda word, *_args, **_kwargs: buttons.get(word),
            _resume_room_if_ready=lambda _: False,
            _click=lambda word, _: clicks.append(word) or True,
            _note_screen=lambda _: None,
            log_info=lambda *_: None,
        )
        QiLingTask._host_cycle_step(task, None)
        self.assertEqual(task._host_phase, "waiting_continue")
        self.assertEqual(clicks, ["放弃结契"])
        QiLingTask._host_cycle_step(task, None)
        self.assertEqual(clicks, ["放弃结契"])
        task._phase_clicked_at -= 1.6
        QiLingTask._host_cycle_step(task, None)
        self.assertEqual(task._host_phase, "waiting_reinvite_dialog")
        self.assertEqual(clicks, ["放弃结契", "点击屏幕继续"])

    def test_abandon_can_go_directly_to_default_invite_dialog(self):
        clicks = []
        label = SimpleNamespace(name="默认邀请队友")
        task = SimpleNamespace(
            _host_phase="battle",
            _phase_clicked_at=time.time(),
            _binding_success_visible=lambda _: False,
            _read_ocr=lambda word, *_args, **_kwargs: label if word in ("放弃结契", "默认邀请队友") else None,
            _click=lambda word, _: clicks.append(word) or True,
            _note_screen=lambda _: None,
            log_info=lambda *_: None,
        )
        QiLingTask._host_cycle_step(task, None)
        self.assertEqual(task._host_phase, "waiting_continue")
        task._phase_clicked_at -= 1.6
        QiLingTask._host_cycle_step(task, None)
        self.assertEqual(task._host_phase, "waiting_reinvite_dialog")
        self.assertEqual(clicks, ["放弃结契"])

    def test_host_can_start_on_binding_failure_page(self):
        clicks = []
        task = SimpleNamespace(
            _host_phase=None,
            _invite_phase=None,
            _binding_success_visible=lambda _: False,
            _read_ocr=lambda word, *_args, **_kwargs: object() if word == "放弃结契" else None,
            _click=lambda word, _: clicks.append(word) or True,
            log_info=lambda *_: None,
        )
        self.assertTrue(QiLingTask._host_scene_action(task, None))
        self.assertEqual(task._host_phase, "waiting_continue")
        self.assertEqual(clicks, ["放弃结契"])

    def test_default_invite_requires_visible_check_before_confirming(self):
        label = SimpleNamespace(name="默认邀请队友", x=150, y=70, width=110, height=20)
        confirm = SimpleNamespace(name="确定", x=150, y=150, width=45, height=20)
        before = np.zeros((220, 320, 3), dtype=np.uint8)
        checked = before.copy()
        checked[76:86, 125:135] = (240, 240, 240)
        self.assertFalse(_checkbox_tick_visible(before, label))
        self.assertTrue(_checkbox_tick_visible(checked, label))
        self.assertTrue(_checkbox_changed(before, checked, label))

        clicks = []
        task = SimpleNamespace(
            _host_phase="waiting_reinvite_dialog",
            _default_invite_before=None,
            _invite_target_name=lambda: "过往的兔耳2",
            _default_invite_box=None,
            _default_checked_frames=0,
            _room_joined_frames=0,
            _read_ocr=lambda word, *_args, **_kwargs: label if word == "默认邀请队友" else confirm,
            _binding_success_visible=lambda _: False,
            _note_screen=lambda _: None,
            _click=lambda text, _: clicks.append(text) or True,
            log_info=lambda *_: None,
        )
        QiLingTask._host_cycle_step(task, before)
        self.assertEqual(clicks, ["默认邀请队友复选框"])
        self.assertEqual(task._host_phase, "checking_default_invite")
        QiLingTask._host_cycle_step(task, before)
        self.assertEqual(clicks, ["默认邀请队友复选框"])
        QiLingTask._host_cycle_step(task, checked)
        self.assertEqual(task._host_phase, "checking_default_invite")
        QiLingTask._host_cycle_step(task, checked)
        self.assertEqual(task._host_phase, "confirm_reinvite")
        QiLingTask._host_cycle_step(task, checked)
        self.assertEqual(clicks, ["默认邀请队友复选框", "续邀确定"])
        self.assertEqual(task._host_phase, "waiting_auto_teammate")

    def test_success_returns_to_support_step(self):
        task = SimpleNamespace(
            _host_phase="battle",
            _invite_phase=None,
            _host_invite_sent=True,
            _pending_transition=None,
            setup_step=HOST_FLOW.index("等待指定队友"),
            _room_joined_frames=1,
            _binding_success_visible=lambda _: True,
            _note_screen=lambda _: None,
            log_info=lambda *_: None,
        )
        task._reset_after_success = lambda: QiLingTask._reset_after_success(task)
        QiLingTask._host_cycle_step(task, None)
        self.assertIsNone(task._host_phase)
        self.assertEqual(task.setup_step, HOST_FLOW.index("求援设置"))
        self.assertFalse(task._host_invite_sent)
        self.assertEqual(task._room_joined_frames, 0)

    def test_success_needs_target_name_and_support_on_same_page(self):
        visible = {"镇墓兽"}
        task = SimpleNamespace(
            _read_ocr=lambda word, *_args, **_kwargs: object() if word in visible else None,
        )
        self.assertFalse(QiLingTask._binding_success_visible(task, None))
        visible.add("求援")
        self.assertTrue(QiLingTask._binding_success_visible(task, None))

    def test_teammate_auto_accepts_prepares_and_continues_without_abandoning(self):
        buttons = {word: SimpleNamespace(name=word) for word in
                   ("队友-选择自动接受", "准备", "点击屏幕继续")}
        showing = {"popup": True, "ready": False, "continue": False}
        clicks = []
        task = SimpleNamespace(
            auto_accept_enabled=False,
            _last_auto_accept_click=0.0,
            _last_teammate_ready_click=0.0,
            _last_continue_click=0.0,
            _threshold=lambda: .8,
            _retry_interval=lambda: 4.0,
            find_one=lambda name, *_args, **_kwargs: buttons[name]
                     if showing["popup"] and name == "队友-选择自动接受" else None,
            _read_ocr=lambda word, *_args, **_kwargs: buttons[word]
                      if ((word == "准备" and showing["ready"])
                          or (word == "点击屏幕继续" and showing["continue"])) else None,
            _detect_screen=lambda _: ("放弃结契", "放弃结契", object(), 1.0),
            _note_screen=lambda _: 0,
            _click=lambda word, _: clicks.append(word) or True,
            log_info=lambda *_: None,
        )
        QiLingTask._teammate_step(task, None)
        self.assertTrue(task.auto_accept_enabled)
        showing.update(popup=False, ready=True)
        QiLingTask._teammate_step(task, None)
        showing.update(ready=False, **{"continue": True})
        QiLingTask._teammate_step(task, None)
        showing["continue"] = False
        QiLingTask._teammate_step(task, None)
        self.assertEqual(clicks, ["队友-选择自动接受", "准备", "点击屏幕继续"])

    def test_binding_page_signature_tolerates_one_missed_word(self):
        self.assertEqual(set(BINDING_PAGE_OCR), {"结契设置", "召唤", "释放"})
        visible = set()
        task = SimpleNamespace(
            ocr=lambda *args, **kwargs: [object()] if kwargs["match"] in visible else [],
            log_warning=lambda *_: None,
        )
        visible.update({"结契设置", "释放"})
        self.assertTrue(QiLingTask._binding_page_visible(task, screenshot("11.png")))
        visible.clear()
        visible.update({"结契设置", "召唤"})
        self.assertTrue(QiLingTask._binding_page_visible(task, screenshot("11.png")))
        visible.clear()
        visible.add("结契设置")
        self.assertFalse(QiLingTask._binding_page_visible(task, screenshot("4.png")))

    def test_binding_page_skips_target_wait_and_seeks_support(self):
        messages = []
        task = SimpleNamespace(
            setup_step=HOST_FLOW.index("选择目标-镇墓兽"),
            _pending_transition=("选择目标-镇墓兽", ("求援设置",), time.time()),
            _pending_last_report=0.0,
            _binding_page_visible=lambda _: True,
            _detect_host_stage=lambda screen, _: (screen, "OCR", object(), 1.0),
            log_info=messages.append,
        )
        QiLingTask._promote_binding_page(task, screenshot("11.png"))
        self.assertEqual(task.setup_step, HOST_FLOW.index("求援设置"))
        self.assertIsNone(task._pending_transition)
        self.assertEqual(QiLingTask._detect_host_screen(task, None)[0], "求援设置")
        self.assertTrue(any("契灵结契页面" in message for message in messages))

        task.setup_step = 0
        QiLingTask._promote_binding_page(task, screenshot("11.png"))
        self.assertEqual(task.setup_step, HOST_FLOW.index("求援设置"))


if __name__ == "__main__":
    unittest.main()
