"""Tải video ghi hình từ Synology Surveillance Station theo khung "giờ trôi qua".

Khung giờ được tính theo thời gian thực: chạy lúc 14:20 ngày 9/9/2026 sẽ tải
video 13:00-14:00 của ngày đó; chạy lúc 00:05 sẽ tải 23:00-00:00 của hôm trước.

Mặc định script đọc `watch_folders.csv` (cột folder,config): với mỗi dòng,
tên camera = tên thư mục cuối cùng và video được tải về đúng thư mục đó để
watch_folders.py (folder watcher) tự nhận và xử lý bằng config tương ứng.

Dùng riêng lẻ:
    python get_video.py                          # khung = giờ trôi qua từ bây giờ
    python get_video.py --at "2026-09-09 14:20"
    python get_video.py --headless
Trong GUI (python -m gui.watch_app) việc tải được hẹn tự động mỗi đầu giờ.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from playwright.sync_api import Download, Page, Playwright, sync_playwright

from watch_folders import load_watch_csv

# Thông số kết nối Surveillance Station (ghi đè được bằng biến môi trường).
DEFAULT_URL = os.getenv(
    "SS_URL",
    "http://172.17.108.143:5000/index.cgi"
    "?launchApp=SYNO.SDS.SurveillanceStation#/signin",
)
DEFAULT_USERNAME = os.getenv("SS_USERNAME", "minh-duc")
DEFAULT_PASSWORD = os.getenv("SS_PASSWORD", "duc20262")

FETCH_STATE_NAME = "fetch_state.json"


def previous_hour_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    """Khung giờ đầy đủ trôi qua gần nhất: tại 14:20 -> (13:00, 14:00).

    Đúng lúc sang giờ/ngày mới (ví dụ 00:00) trả về khung 23:00-00:00 hôm trước.
    """
    now = now or datetime.now()
    end = now.replace(minute=0, second=0, microsecond=0)
    return end - timedelta(hours=1), end


def click_download_dialog_button(page: Page, index: int) -> None:
    """Click nút Download thật bên trong hộp thoại xác nhận tải xuống."""

    # Ưu tiên phần text của nút nằm trong dialog đang hiển thị
    candidates = [
        lambda: page.locator(
            ".x-window:visible .x-btn-text"
        ).filter(has_text="Download").last,

        lambda: page.locator(
            ".syno-window:visible .x-btn-text"
        ).filter(has_text="Download").last,

        lambda: page.locator(
            ".x-window:visible .syno-ux-button"
        ).filter(has_text="Download").last,

        lambda: page.locator(
            ".syno-window:visible .syno-ux-button"
        ).filter(has_text="Download").last,

        lambda: page.get_by_role(
            "button",
            name="Download",
            exact=True,
        ).last,

        lambda: page.get_by_text(
            "Download",
            exact=True,
        ).last,
    ]

    errors = []

    for number, make_locator in enumerate(candidates, start=1):
        button = make_locator()

        try:
            button.wait_for(
                state="visible",
                timeout=5000,
            )

            button.scroll_into_view_if_needed(
                timeout=3000
            )

            # Click phần tử nút thật
            button.click(
                timeout=5000,
                force=True,
            )

            print(
                f"[click] nut Download trong dialog video "
                f"{index + 1}: dùng locator #{number}"
            )
            return

        except Exception as exc:
            errors.append(
                f"#{number}: {type(exc).__name__}"
            )

    raise RuntimeError(
        f"Không click được nút Download trong dialog "
        f"của video {index + 1}. "
        f"Lỗi: {'; '.join(errors)}"
    )
def get_recording_rows(page: Page, camera_name: str):
    """Lấy các phần tử chứa chính xác tên camera trong kết quả tìm kiếm."""

    matches = page.get_by_text(
        camera_name,
        exact=True,
    )

    visible_items = []

    for index in range(matches.count()):
        item = matches.nth(index)

        try:
            if item.is_visible():
                visible_items.append(item)
        except Exception:
            continue

    return visible_items


def right_click_recording(
    page: Page,
    camera_name: str,
    index: int,
    total: int,
) -> None:
    """Click phải vào kết quả video thứ index."""

    last_error = None

    for attempt in range(1, 6):
        try:
            items = get_recording_rows(page, camera_name)

            if index >= len(items):
                raise RuntimeError(
                    f"Không tìm thấy video thứ {index + 1}. "
                    f"Hiện có {len(items)} video hiển thị."
                )

            item = items[index]
            item.wait_for(state="visible", timeout=5000)
            item.scroll_into_view_if_needed(timeout=5000)

            print(
                f"[download] click phải video "
                f"{index + 1}/{total}, lần thử {attempt}"
            )

            # Click trên đúng phần tử chữ camera.
            # Sự kiện chuột phải sẽ nổi lên dòng kết quả bên ngoài.
            item.click(
                button="right",
                timeout=5000,
                force=True,
            )

            return

        except Exception as exc:
            last_error = exc
            print(
                f"[download] click video lần {attempt} lỗi: "
                f"{type(exc).__name__}: {exc}"
            )
            page.wait_for_timeout(1000)

    raise RuntimeError(
        f"Không thể click phải video thứ {index + 1} sau 5 lần thử."
    ) from last_error

def click_checkbox_near_text(page: Page, text: str) -> None:
    """Click checkbox nằm gần chữ Date hoặc Time."""

    label = page.get_by_text(text, exact=True).first
    label.wait_for(state="visible", timeout=5000)

    checkbox = None

    # Tìm checkbox trong các cấp phần tử cha của chữ Date/Time
    for parent_level in range(1, 5):
        parent = label.locator(
            "xpath=" + "/.." * parent_level
        )

        candidate = parent.locator(
            "input[type='checkbox'], "
            ".syno-ux-checkbox-icon, "
            ".x-form-checkbox"
        ).first

        if candidate.count() == 0:
            continue

        try:
            candidate.wait_for(state="visible", timeout=1000)
            checkbox = candidate
            break
        except Exception:
            pass

    if checkbox is None:
        raise RuntimeError(
            f"Không tìm thấy checkbox nằm bên cạnh chữ '{text}'."
        )

    print(f"[checkbox] tìm thấy checkbox của '{text}'")

    # Thử click bình thường trước
    try:
        checkbox.click(timeout=3000)
    except Exception:
        # Nếu checkbox bị lớp giao diện che, thử click cưỡng chế
        try:
            checkbox.click(timeout=3000, force=True)
        except Exception:
            # Cuối cùng gọi click bằng JavaScript
            checkbox.evaluate("el => el.click()")

    print(f"[checkbox] đã click checkbox của '{text}'")

def tick_camera_checkbox(page: Page, camera_name: str) -> None:
    """Tick checkbox ở cột đầu tiên của dòng chứa tên camera trong grid ExtJS.

    Xác định dòng (.x-grid3-row) đang hiển thị có chứa tên camera, rồi click
    vào icon checkbox trong ô .x-grid3-td-check. Sau mỗi lần click, so sánh
    class của icon trước/sau để chắc chắn trạng thái thật sự thay đổi; nếu
    không sẽ thử lần lượt: click đè (force), click cả ô checkbox, click JS.
    """
    # Chỉ xét các dòng ĐANG HIỂN THỊ để không trúng các dòng ẩn mà cơ chế
    # virtual scrolling của ExtJS vẫn giữ lại trong DOM.
    matched = page.locator(".x-grid3-row:visible").filter(has_text=camera_name)
    row = matched.first

    try:
        row.wait_for(state="visible", timeout=10000)
    except Exception as exc:
        raise RuntimeError(
            f"Không tìm thấy dòng nào đang hiển thị chứa '{camera_name}'."
        ) from exc

    print(f"[checkbox] số dòng hiển thị khớp: {matched.count()}")

    icon = row.locator(".x-grid3-td-check .syno-ux-checkbox-icon").first
    cell = row.locator(".x-grid3-td-check").first

    if icon.count() == 0:
        # DOM khác giả định (không có .syno-ux-checkbox-icon) -> click cả ô
        cell.click(timeout=5000)
        print("[checkbox] không thấy icon, đã click trực tiếp ô .x-grid3-td-check")
        return

    page.mouse.move(0, 0)  # gỡ trạng thái hover để đọc class "sạch"
    before = icon.get_attribute("class") or ""
    print(f"[checkbox] class trước khi click: {before!r}")
    print(f"[checkbox] toạ độ icon: {icon.bounding_box()}")  # None = icon đang ẩn

    attempts = (
        ("click icon", lambda: icon.click(timeout=3000)),
        ("click icon (force)", lambda: icon.click(timeout=3000, force=True)),
        ("click ô .x-grid3-td-check", lambda: cell.click(timeout=3000)),
        ("click icon bằng JS", lambda: icon.evaluate("el => el.click()")),
    )

    for label, action in attempts:
        try:
            action()
        except Exception as exc:
            print(f"[checkbox] '{label}' lỗi: {type(exc).__name__}")
            continue

        page.mouse.move(0, 0)
        after = icon.get_attribute("class") or ""
        if after != before:
            print(f"[checkbox] OK qua '{label}': {before!r} -> {after!r}")
            return
        print(f"[checkbox] '{label}' không làm đổi trạng thái checkbox")

    raise RuntimeError(
        "Đã thử mọi cách click nhưng checkbox không đổi trạng thái. "
        f"class icon: {before!r}. Hãy mở page.pause() và dùng 'Pick locator' "
        "chọn đúng checkbox để xem class/bộ chọn thật của nó."
    )


def click_first_working(
    page: Page,
    make_locators,
    step: str,
    **click_kwargs,
) -> None:
    """Thử lần lượt các locator (ổn định trước, dự phòng sau) rồi click.

    make_locators: danh sách hàm, mỗi hàm trả về một Locator. Locator nào
    tìm thấy và click được sẽ được dùng; nếu tất cả đều thất bại thì báo
    lỗi kèm tên bước để dễ xử lý.
    """
    problems = []
    for index, make_locator in enumerate(make_locators, start=1):
        locator = make_locator()
        try:
            locator.first.wait_for(state="visible", timeout=2000)
            # timeout click ngắn: ứng viên hụt chỉ tốn ~3s thay vì 30s mặc
            # định - tránh khiến người dùng tưởng script bị treo.
            locator.first.click(timeout=3000, **click_kwargs)
            print(f"[click] {step}: dùng locator #{index}")
            return
        except Exception as exc:
            problems.append(f"#{index}: {type(exc).__name__}")
    # In luôn gợi ý các phần tử đang hiển thị để tự chẩn đoán ngay, không cần
    # bật GET_VIDEO_DEBUG (ID có thể đã đổi, hoặc ứng dụng chưa mở như kỳ vọng).
    print(
        f"\n[click] Bước '{step}' thất bại. "
        "Các phần tử ĐANG HIỂN THỊ trên màn hình lúc này:"
    )
    dump_locator_suggestions(page)
    raise RuntimeError(
        f"Bước '{step}' thất bại - không có locator nào dùng được "
        f"({'; '.join(problems)}). Xem mục 'GỢI Ý BỘ CHỌN ỔN ĐỊNH' in phía trên: "
        "tìm đúng nút của bước này rồi thêm locator theo chữ của nó vào ĐẦU "
        "danh sách locator của bước (xem ví dụ trong comment của get_video.py). "
        "Hoặc chạy lại với GET_VIDEO_DEBUG=1 để mở Inspector tự xem DOM."
    )


def _print_open_panels(page: Page) -> None:
    """In nhanh các panel/layer đang mở để tìm ra class picker thật."""
    try:
        rows = page.evaluate(
            """() => Array.from(document.querySelectorAll(
                    '.x-layer, .x-date-picker, .x-combo-list, .x-menu, .x-tip,' +
                    ' [class*="picker"], [class*="combo-list"]'
                ))
                .filter(el => el.getClientRects().length > 0)
                .slice(0, 15)
                .map(el => ({id: el.id, cls: el.className}))"""
        )
    except Exception as exc:
        print(f"[probe] không lấy được panel đang mở: {type(exc).__name__}")
        return
    if not rows:
        print("[probe] không có panel/layer nào đang mở trên màn hình")
        return
    print("[probe] các panel/layer ĐANG MỞ (tìm class picker thật ở đây):")
    for r in rows:
        print(f"  - id={r['id']!r} class={r['cls']!r}")


def click_unless_visible(
    page: Page, open_indicator: str, make_locators, step: str
) -> None:
    """Giống click_first_working, TRỪ KHI open_indicator đã hiển thị.

    Chờ tối đa ~2.5s cho lịch/dropdown xuất hiện (tránh đua với hiệu ứng
    mở) rồi mới quyết định: thấy -> bỏ qua click; không thấy -> in probe
    các panel đang mở rồi mới click.
    """
    try:
        page.locator(open_indicator).first.wait_for(
            state="visible", timeout=2500
        )
        print(f"[click] {step}: '{open_indicator}' đã hiển thị -> bỏ qua click")
        return
    except Exception:
        _print_open_panels(page)
    click_first_working(page, make_locators, step)


def set_search_range(
    page: Page,
    begin_dt: datetime,
    end_dt: datetime,
) -> None:
    """Gán TRỰC TIẾP giá trị ngày/giờ vào dialog Search qua API ExtJS.

    Thay vì click mở lịch/dropdown (input bị lớp trigger chặn, picker không
    mở khi bị automation click), lấy component của từng ô qua Ext.getCmp
    rồi gọi setValue -> chắc chắn, không phụ thuộc phiên đăng nhập.
    Thứ tự ô theo DOM: datefield/timefield HIỂN THỊ thứ 0 = bắt đầu,
    thứ 1 = kết thúc. Ô ngày bắt đầu lấy ngày của begin_dt, ô ngày kết thúc
    lấy ngày của end_dt (khác nhau khi khung giờ vắt qua nửa đêm,
    ví dụ 23:00-00:00).
    """
    result = page.evaluate(
        """(vals) => {
            if (!window.Ext || !Ext.getCmp) {
                return {error: 'Không tìm thấy ExtJS (window.Ext) trên trang'};
            }
            const pick = (cls) => Array.from(
                document.querySelectorAll('input.' + cls)
            ).filter(el => el.getClientRects().length > 0);
            const dates = pick('syno-ux-datefield');
            const times = pick('syno-ux-timefield');
            if (dates.length < 2 || times.length < 2) {
                return {
                    error: 'Không đủ ô ngày/giờ hiển thị',
                    soDate: dates.length,
                    soTime: times.length,
                };
            }
            const setOne = (el, val) => {
                const comp = Ext.getCmp(el.id);
                if (!comp || !comp.setValue) {
                    return {ok: false, id: el.id, why: 'no-component'};
                }
                comp.setValue(val);
                return {ok: !!el.value, id: el.id, value: el.value};
            };
            const begin = new Date(
                vals.begin.year, vals.begin.month - 1, vals.begin.day
            );
            const end = new Date(vals.end.year, vals.end.month - 1, vals.end.day);
            return {
                beginDate: setOne(dates[0], begin),
                endDate: setOne(dates[1], end),
                beginTime: setOne(times[0], vals.beginTime),
                endTime: setOne(times[1], vals.endTime),
            };
        }""",
        {
            "begin": {
                "day": begin_dt.day,
                "month": begin_dt.month,
                "year": begin_dt.year,
            },
            "end": {
                "day": end_dt.day,
                "month": end_dt.month,
                "year": end_dt.year,
            },
            "beginTime": begin_dt.strftime("%H:%M"),
            "endTime": end_dt.strftime("%H:%M"),
        },
    )
    print(f"[datetime] kết quả gán qua ExtJS: {result}")
    if not isinstance(result, dict) or result.get("error"):
        raise RuntimeError(f"Gán ngày/giờ thất bại: {result}")
    for key in ("beginDate", "endDate", "beginTime", "endTime"):
        item = result.get(key) or {}
        if not item.get("ok"):
            raise RuntimeError(f"Gán {key} thất bại: {item}")


def dump_locator_suggestions(page: Page) -> None:
    """In các nút/trường đang hiển thị (text, id, class) để tìm bộ chọn ổn định
    thay cho ext-gen/ext-comp. Dùng kèm biến môi trường GET_VIDEO_DEBUG=1.
    """
    print("\n===== GỢI Ý BỘ CHỌN ỔN ĐỊNH (GET_VIDEO_DEBUG) =====")
    groups = {
        "nut bam": ".x-btn-text, .syno-ux-button, button, [role=button]",
        "nut trong dialog (.x-window)": (
            ".x-window .x-btn-text, .x-window .syno-ux-button, "
            ".syno-window .x-btn-text, .syno-window .syno-ux-button"
        ),
        "tieu de cua so": ".x-window-header-text, .x-window-title, .syno-window-title",
        "panel/layer dang mo": (
            '.x-layer, .x-date-picker, .x-combo-list, .x-menu, .x-tip,'
            ' [class*="picker"]'
        ),
        "checkbox": (
            ".syno-ux-checkbox-icon, input[type=checkbox], .x-form-checkbox"
        ),
        "truong nhap/combobox": (
            "input.x-form-text, .syno-ux-datefield input, "
            ".syno-ux-timefield input, .syno-ux-combobox input"
        ),
        "muc menu": ".x-menu-item-text",
    }
    try:
        for label, selector in groups.items():
            try:
                rows = page.evaluate(
                    """(sel) => {
                        const all = document.querySelectorAll(sel);
                        const out = [];
                        for (const el of all) {
                            if (out.length >= 40) break;
                            if (!el.getClientRects().length) continue;
                            out.push({
                                text: ((el.innerText || el.value || '') + '')
                                    .trim()
                                    .slice(0, 60),
                                id: el.id,
                                cls: el.className,
                            });
                        }
                        return {total: all.length, rows: out};
                    }""",
                    selector,
                )
            except Exception as exc:
                print(f"--- {label}: không lấy được ({type(exc).__name__}) ---")
                continue
            print(f"--- {label} ({rows['total']} phần tử, chỉ in phần hiển thị) ---")
            for i, r in enumerate(rows["rows"]):
                text = r["text"].replace("\n", " | ")
                print(f"  [{i}] text={text!r} id={r['id']!r} class={r['cls']!r}")
    except Exception as exc:
        print(f"  (dừng lấy gợi ý: {type(exc).__name__} - trang có thể đã đóng)")
    print("===== KẾT THÚC GỢI Ý =====\n")


def debug_checkpoint(page: Page, label: str) -> None:
    """Ở chế độ debug (GET_VIDEO_DEBUG=1): in gợi ý bộ chọn rồi mở Inspector."""
    if os.getenv("GET_VIDEO_DEBUG"):
        print(f"\n##### DEBUG CHECKPOINT: {label} #####")
        dump_locator_suggestions(page)


def run(
    playwright: Playwright,
    camera_name: str,
    download_dir: str | Path,
    begin_dt: datetime,
    end_dt: datetime,
    *,
    headless: bool = False,
    url: str = DEFAULT_URL,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
) -> list[Path]:
    """Tải toàn bộ video ghi hình của một camera trong khung [begin_dt, end_dt).

    Trả về danh sách các tệp đã lưu vào download_dir.
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Mở Microsoft Edge
    browser = playwright.chromium.launch(
        channel="msedge",
        headless=headless
    )

    context = browser.new_context(
        accept_downloads=True
    )

    page = context.new_page()

    page.goto(url)

    page.get_by_role("textbox", name="Username").fill(username)
    page.get_by_role("button", name="Sign In").click()

    page.get_by_role("textbox", name="Password").fill(password)
    page.get_by_role("button", name="Sign In").click()

    page.get_by_role(
        "link",
        name="Recording Recording"
    ).get_by_role("img").click()
    debug_checkpoint(page, "ngay sau khi mo ung dung Recording")

    # Bước 1: nút "Search" trên thanh công cụ của Recording (đã xác nhận).
    # Ưu tiên locator theo CHỮ nút -> ổn định giữa các phiên; ID cũ chỉ dự phòng.
    click_first_working(
        page,
        [
            lambda: page.locator(".syno-ux-button:visible", has_text="Search"),
            lambda: page.locator(".x-btn-text:visible", has_text="Search"),
            lambda: page.get_by_text("Search", exact=True),
            lambda: page.locator("#ext-gen1407"),
        ],
        "nut Search (cu: #ext-gen1407)",
    )

    # Bước 2: cú click thứ hai ngay sau khi bấm Search. Nếu đây là mục trong
    # menu xổ ra (ví dụ "Advanced Search") thì 2 locator đầu sẽ khớp; nếu mọi
    # ứng viên đều hụt, script sẽ in gợi ý phần tử trên màn hình để bổ sung.
    click_first_working(
        page,
        [
            lambda: page.locator(".x-menu-item-text:visible", has_text="Search"),
            lambda: page.locator(".x-menu-item-text:visible", has_text="Advanced"),
            lambda: page.locator("#ext-gen283"),
        ],
        "buoc 2 sau khi mo Search (cu: #ext-gen283)",
    )
    

    page.locator(
        ".syno-ux-checkbox-icon.ss-chkbox-check-normal"
    ).first.click()

    # Tick chọn checkbox của dòng có tên camera
    tick_camera_checkbox(page, camera_name)

    debug_checkpoint(
        page, "sau khi tick checkbox camera (truoc khi chon ngay/gio)"
    )

    click_checkbox_near_text(page, "Date")

    # Tick checkbox Time
    click_checkbox_near_text(page, "Time")



    # ---- Chọn khoảng ngày/giờ: gán TRỰC TIẾP qua API ExtJS ----
    # Click mở lịch/dropdown trên Synology không đáng tin (input bị lớp
    # trigger chặn, picker không mở khi automation click) -> dùng Ext.getCmp
    # + setValue. Ô ngày bắt đầu = ngày của begin_dt, ô ngày kết thúc =
    # ngày của end_dt (khác nhau khi khung giờ vắt qua nửa đêm).
    set_search_range(page, begin_dt, end_dt)


    # (việc chọn giờ đã gộp vào set_search_range phía trên)

    # Nút xác nhận của dialog: chỉ xét nút nằm trong FOOTER của cửa sổ dialog
    # (tránh nhầm với nút Search trên thanh công cụ phía sau).
    # Click nút Search màu xanh ở phía dưới dialog
    search_button = page.get_by_role(
        "button",
        name="Search",
        exact=True,
    ).last

    search_button.wait_for(state="visible", timeout=5000)
    search_button.click(timeout=5000)

    print("[click] đã bấm nút Search màu xanh ở cuối dialog")

    def download_all_recordings() -> None:
        """Tải toàn bộ video trong kết quả tìm kiếm."""

        # Chờ danh sách kết quả hiển thị hoàn toàn
        page.wait_for_timeout(2000)

        items = get_recording_rows(page, camera_name)
        total = len(items)

        if total == 0:
            raise RuntimeError(
                f"Không tìm thấy video nào có tên chính xác '{camera_name}'."
            )

        print(f"[download] tìm thấy {total} video cần tải")

        for index in range(total):
            # Click phải video hiện tại
            right_click_recording(
                page,
                camera_name,
                index,
                total,
            )

            # Menu chuột phải có thể dùng link hoặc phần tử text
            download_menu = page.get_by_text(
                "Download",
                exact=True,
            ).last

            download_menu.wait_for(
                state="visible",
                timeout=5000,
            )
            download_menu.click(timeout=5000)

            # Xác nhận tải xuống
            with page.expect_download(timeout=30000) as download_info:
                click_download_dialog_button(page, index)

            download = download_info.value

            target = save_download(download, download_dir, camera_name, index)
            saved.append(target)

            print(f"[download] đã lưu file tại: {target}")

            # Đợi giao diện trở lại ổn định trước khi lấy video tiếp theo
            page.wait_for_timeout(1000)

    download_all_recordings()

    context.close()
    browser.close()

    return saved


def save_download(
    download: Download,
    download_dir: Path,
    camera_name: str,
    index: int,
) -> Path:
    """Lưu tệp tải về, ưu tiên tên gốc của Surveillance Station.

    Tên gốc thường chứa mốc thời gian (ví dụ CA927-...-20260909-131530.mp4)
    nên hữu ích khi tra cứu. Nếu tệp trùng tên đã có thì thêm hậu tố
    _2, _3, ... để không ghi đè.
    """
    suggested = (download.suggested_filename or "").strip()
    if suggested:
        stem = Path(suggested).stem
        suffix = Path(suggested).suffix or ".mp4"
    else:
        stem = f"{camera_name}_{index + 1:02d}"
        suffix = ".mp4"
    stem = "".join(c if c not in '\\/:*?"<>|' else "_" for c in stem)

    target = download_dir / f"{stem}{suffix}"
    duplicate = 2
    while target.exists():
        target = download_dir / f"{stem}_{duplicate}{suffix}"
        duplicate += 1

    download.save_as(str(target))
    return target


def fetch_one_camera(
    camera_name: str,
    download_dir: str | Path,
    begin_dt: datetime,
    end_dt: datetime,
    *,
    headless: bool = False,
    url: str = DEFAULT_URL,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
) -> list[Path]:
    """Mở trình duyệt, đăng nhập và tải video của MỘT camera trong khung giờ."""
    with sync_playwright() as playwright:
        return run(
            playwright,
            camera_name,
            download_dir,
            begin_dt,
            end_dt,
            headless=headless,
            url=url,
            username=username,
            password=password,
        )


def load_fetch_state(path: str | Path) -> dict:
    """Đọc tệp trạng thái đã tải (cấu trúc: {"cameras": {tên: mốc giờ}})."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("cameras"), dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"cameras": {}}


def save_fetch_state(path: str | Path, state: dict) -> None:
    """Ghi trạng thái (atomic); lỗi chỉ được log, không làm gãy vòng lặp."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError as e:
        print(f"WARNING: could not save fetch state {path}: {e}")


def fetch_all_from_csv(
    csv_path: str | Path,
    begin_dt: datetime,
    end_dt: datetime,
    *,
    headless: bool = False,
    should_stop: Callable[[], bool] | None = None,
    log: Callable[[str], None] = print,
    state_path: str | Path | None = None,
) -> dict:
    """Tải video của khung [begin_dt, end_dt) cho TỪNG camera trong tệp CSV.

    Tên camera = tên thư mục cuối trong cột `folder`; video được tải về
    chính thư mục đó để folder watcher (watch_folders.py) tự nhận và xử lý
    tiếp bằng config của dòng tương ứng. Camera bị lỗi KHÔNG chặn các
    camera phía sau. Khung giờ đã tải thành công được ghi vào state file
    (mặc định outputs/fetch_state.json) để lần chạy sau không tải trùng.
    """
    entries, errors = load_watch_csv(csv_path)
    for error in errors:
        log(f"[fetch] CẢNH BÁO CSV: {error}")

    state_path = Path(state_path) if state_path else Path("outputs") / FETCH_STATE_NAME
    state = load_fetch_state(state_path)
    window_key = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    window_label = f"{begin_dt:%H:%M}-{end_dt:%H:%M} ngày {end_dt:%d/%m/%Y}"

    results: list[dict] = []

    for entry in entries:
        folder = Path(entry["folder"])
        camera = folder.name

        if should_stop is not None and should_stop():
            log("[fetch] Đã dừng theo yêu cầu.")
            break

        if state["cameras"].get(camera) == window_key:
            log(f"[fetch] {camera}: khung {window_label} đã tải trước đó, bỏ qua.")
            results.append(
                {"camera": camera, "status": "skipped", "folder": str(folder)}
            )
            continue

        log(f"[fetch] {camera}: đang tải khung {window_label} ...")
        try:
            saved = fetch_one_camera(camera, folder, begin_dt, end_dt, headless=headless)
            state["cameras"][camera] = window_key
            save_fetch_state(state_path, state)
            log(f"[fetch] {camera}: đã lưu {len(saved)} file.")
            results.append(
                {
                    "camera": camera,
                    "status": "ok",
                    "count": len(saved),
                    "folder": str(folder),
                }
            )
        except Exception as exc:
            log(f"[fetch] {camera}: LỖI - {type(exc).__name__}: {exc}")
            results.append(
                {
                    "camera": camera,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "folder": str(folder),
                }
            )

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "error")
    log(f"[fetch] Hoàn tất: {ok} thành công, {skipped} đã tải trước đó, {failed} lỗi.")
    return {
        "total": len(entries),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "results": results,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tải video của khung 'giờ trôi qua' từ Surveillance Station cho "
            "mọi camera khai báo trong watch_folders.csv."
        )
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="watch_folders.csv",
        help="Tệp CSV (cột folder,config; mặc định: watch_folders.csv).",
    )
    parser.add_argument(
        "--at",
        default=None,
        metavar='"YYYY-MM-DD HH:MM"',
        help="Mô phỏng thời điểm chạy để kiểm thử (mặc định: bây giờ).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Chạy trình duyệt ẩn, không mở cửa sổ Edge.",
    )
    parser.add_argument(
        "--state",
        default=None,
        help=f"Tệp trạng thái đã tải (mặc định: outputs/{FETCH_STATE_NAME}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    now = None
    if args.at:
        try:
            now = datetime.strptime(args.at, "%Y-%m-%d %H:%M")
        except ValueError:
            print(f"--at cần định dạng 'YYYY-MM-DD HH:MM', nhận được: {args.at!r}")
            return 2

    begin_dt, end_dt = previous_hour_window(now)
    print(f"[fetch] Khung video: {begin_dt:%d/%m/%Y %H:%M} - {end_dt:%H:%M}")

    summary = fetch_all_from_csv(
        args.csv,
        begin_dt,
        end_dt,
        headless=args.headless,
        state_path=args.state,
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())