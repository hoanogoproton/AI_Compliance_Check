import os
from pathlib import Path

from playwright.sync_api import Page, Playwright, sync_playwright

CAMERA_NAME = "CA927-FB-RAI7-No3"


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
    day: int,
    month: int,
    year: int,
    begin_time: str,
    end_time: str,
) -> None:
    """Gán TRỰC TIẾP giá trị ngày/giờ vào dialog Search qua API ExtJS.

    Thay vì click mở lịch/dropdown (input bị lớp trigger chặn, picker không
    mở khi bị automation click), lấy component của từng ô qua Ext.getCmp
    rồi gọi setValue -> chắc chắn, không phụ thuộc phiên đăng nhập.
    Thứ tự ô theo DOM: datefield/timefield HIỂN THỊ thứ 0 = bắt đầu,
    thứ 1 = kết thúc (cả hai ô ngày được gán cùng một ngày như bản ghi gốc).
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
            const d = new Date(vals.year, vals.month - 1, vals.day);
            return {
                beginDate: setOne(dates[0], d),
                endDate: setOne(dates[1], d),
                beginTime: setOne(times[0], vals.beginTime),
                endTime: setOne(times[1], vals.endTime),
            };
        }""",
        {
            "day": day,
            "month": month,
            "year": year,
            "beginTime": begin_time,
            "endTime": end_time,
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


def run(playwright: Playwright) -> None:
    # Thư mục lưu file tải xuống
    download_dir = Path(r"D:\Video\CA927-FB-RAI7-No3")
    download_dir.mkdir(parents=True, exist_ok=True)

    # Mở Microsoft Edge
    browser = playwright.chromium.launch(
        channel="msedge",
        headless=False
    )

    context = browser.new_context(
        accept_downloads=True
    )

    page = context.new_page()

    page.goto(
        "http://172.17.108.143:5000/index.cgi"
        "?launchApp=SYNO.SDS.SurveillanceStation#/signin"
    )

    page.get_by_role("textbox", name="Username").fill("minh-duc")
    page.get_by_role("button", name="Sign In").click()

    page.get_by_role("textbox", name="Password").fill("duc20262")
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
    tick_camera_checkbox(page, CAMERA_NAME)

    debug_checkpoint(
        page, "sau khi tick checkbox camera (truoc khi chon ngay/gio)"
    )
    page.pause()  # tạm dừng để người dùng kiểm tra giao diện trước khi chọn ngày/giờ

    # ---- Chọn khoảng ngày/giờ: gán TRỰC TIẾP qua API ExtJS ----
    # Click mở lịch/dropdown trên Synology không đáng tin (input bị lớp
    # trigger chặn, picker không mở khi automation click) -> dùng Ext.getCmp
    # + setValue. Cả 2 ô ngày = cùng một ngày (như bản ghi gốc: chọn ngày 8
    # ở cả 2 lịch); giờ bắt đầu/kết thúc = 13:00/14:00.
    set_search_range(
        page,
        day=8,
        month=9,
        year=2026,
        begin_time="13:00",
        end_time="14:00",
    )

    # (việc chọn giờ đã gộp vào set_search_range phía trên)

    # Nút xác nhận của dialog: chỉ xét nút nằm trong FOOTER của cửa sổ dialog
    # (tránh nhầm với nút Search trên thanh công cụ phía sau).
    click_first_working(
        page,
        [
            lambda: page.locator(
                ".x-window .syno-ux-button:visible, "
                ".syno-window .syno-ux-button:visible",
                has_text="Search",
            ),
            lambda: page.locator(
                ".x-window .syno-ux-button:visible, "
                ".syno-window .syno-ux-button:visible",
                has_text="OK",
            ),
            lambda: page.locator(
                ".x-window .x-btn-text:visible, "
                ".syno-window .x-btn-text:visible",
                has_text="Search",
            ),
            lambda: page.locator(
                ".x-window .x-btn-text:visible, "
                ".syno-window .x-btn-text:visible",
                has_text="OK",
            ),
            lambda: page.locator("#ext-gen206"),
        ],
        "nut xac nhan dialog (cu: #ext-gen206)",
    )

    def download_recording(
        recording_id: str, save_name: str, ok_fallback: str
    ) -> None:
        """Click phải node ghi hình rồi tải file về.

        recording_id: mã số bản ghi trong CSDL Surveillance Station - là
        PHẦN ĐUÔI của id node (ext-comp-1252-0_4703634 -> '4703634').
        Phần đuôi này ổn định giữa các phiên, chỉ phần đầu 'ext-comp-...'
        là tự sinh, nên dùng bộ chọn [id$="_<recording_id>"].
        """
        click_first_working(
            page,
            [
                lambda: page.locator(f'[id$="_{recording_id}"]').filter(
                    has_text=CAMERA_NAME
                ),
                lambda: page.locator(f"#ext-comp-1252-0_{recording_id}"),
            ],
            f"node ghi hình {recording_id} (click phải)",
            button="right",
        )

        page.get_by_role("link", name="Download").click()

        with page.expect_download() as download_info:
            click_first_working(
                page,
                [
                    lambda: page.locator(".syno-ux-button:visible", has_text="OK"),
                    lambda: page.locator(".x-btn-text:visible", has_text="OK"),
                    lambda: page.locator(ok_fallback),
                ],
                f"nut OK cua hop thoai download ({ok_fallback})",
            )

        download = download_info.value
        target = download_dir / (
            save_name + Path(download.suggested_filename).suffix
        )
        download.save_as(str(target))
        print(f"Đã lưu file tại: {target}")

    # Tải file thứ nhất
    download_recording("4703634", f"{CAMERA_NAME}_1", "#ext-gen2525")

    # Tải file thứ hai
    download_recording("4703575", f"{CAMERA_NAME}_2", "#ext-gen2610")

    context.close()
    browser.close()


if __name__ == "__main__":
    with sync_playwright() as playwright:
        run(playwright)