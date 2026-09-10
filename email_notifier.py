"""
Email notifier for the folder watcher: sends video processing results to the
factory email service over a raw TCP socket.

Message format (single line, UTF-8, `|` as field delimiter):

    [SendEmail_KTTT]{factory}|{subject}|{body}

The results body is a single-line professional HTML document built by
``build_results_html``; ``|`` never appears in subject or body.

Edit the constants below to point at a different server or factory.
"""

from __future__ import annotations

import html
import socket
import threading
from datetime import datetime

SERVER_IP = "172.17.108.169"
SERVER_PORT = 1111
FACTORY = "PM6"
SOCKET_TIMEOUT = 10  # seconds

_DASH = "\u2014"
_ACCENT = "#0b4f9e"
_CELL = "padding:8px 10px;border-bottom:1px solid #e8edf3;"
_CELL_CENTER = _CELL + "text-align:center;"

_TEMPLATE = (
    '<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1"></head>'
    '<body style="margin:0;padding:0;background-color:#f2f5f9;">'
    '<table role="presentation" width="100%" cellpadding="0" cellspacing="0"'
    ' style="background-color:#f2f5f9;padding:24px 8px;'
    'font-family:Arial,Helvetica,sans-serif;">'
    "<tr><td align=\"center\">"
    '<table role="presentation" width="640" cellpadding="0" cellspacing="0"'
    ' style="width:640px;max-width:100%;background-color:#ffffff;'
    'border:1px solid #e3e8ee;border-radius:8px;overflow:hidden;">'
    '<tr><td style="background-color:#0b4f9e;padding:22px 32px;">'
    '<div style="color:#ffffff;font-size:20px;font-weight:bold;'
    'letter-spacing:0.3px;">BÁO CÁO KẾT QUẢ XỬ LÝ VIDEO</div>'
    '<div style="color:#bcd3f0;font-size:13px;margin-top:6px;">'
    "Hệ thống nhận diện hành vi - Nhà máy __FACTORY__</div>"
    "</td></tr>"
    '<tr><td style="padding:26px 32px 24px;">'
    '<p style="margin:0 0 12px;font-size:14px;color:#37414d;">'
    "Kính gửi Quý anh/chị,</p>"
    '<p style="margin:0 0 18px;font-size:14px;color:#37414d;line-height:1.6;">'
    'Hệ thống đã xử lý xong video <b style="color:#0b4f9e;">__VIDEO__</b> '
    'và ghi nhận <b style="color:#c0392b;">__TOTAL__ sự kiện</b>. '
    "Danh sách chi tiết:</p>"
    '<table width="100%" cellpadding="0" cellspacing="0" role="presentation"'
    ' style="border-collapse:collapse;font-size:13px;color:#37414d;">'
    '<tr style="background-color:#0b4f9e;color:#ffffff;font-size:12px;">'
    '<th style="padding:9px 10px;text-align:center;font-weight:bold;">STT</th>'
    '<th style="padding:9px 10px;text-align:center;font-weight:bold;">Mã sự kiện</th>'
    '<th style="padding:9px 10px;text-align:left;font-weight:bold;">Hành vi</th>'
    '<th style="padding:9px 10px;text-align:center;font-weight:bold;">Bắt đầu (s)</th>'
    '<th style="padding:9px 10px;text-align:center;font-weight:bold;">Kết thúc (s)</th>'
    '<th style="padding:9px 10px;text-align:center;font-weight:bold;">Thời lượng (s)</th>'
    "</tr>"
    "__ROWS__"
    "</table>"
    '<table width="100%" cellpadding="0" cellspacing="0" role="presentation"'
    ' style="margin-top:18px;border-collapse:collapse;font-size:13px;'
    'color:#37414d;">'
    '<tr><td style="padding:3px 0;width:150px;color:#69758a;">Thời gian xử lý</td>'
    '<td style="padding:3px 0;font-weight:bold;">__TIME__</td></tr>'
    '<tr><td style="padding:3px 0;color:#69758a;">Thư mục kết quả</td>'
    '<td style="padding:3px 0;font-family:Consolas,monospace;font-size:12px;">'
    "__FOLDER__</td></tr>"
    "</table>"
    "</td></tr>"
    '<tr><td style="background-color:#f0f3f7;padding:14px 32px;font-size:12px;'
    'color:#8a94a3;line-height:1.5;">Đây là email tự động từ hệ thống nhận '
    "diện hành vi - Nhà máy __FACTORY__. Vui lòng không trả lời email này."
    "</td></tr>"
    "</table></td></tr></table></body></html>"
)


def send_email(subject: str, body: str) -> tuple[str, str]:
    """Send one email request and return ``(message, response)``.

    Blocking call: it can wait up to ``SOCKET_TIMEOUT`` seconds. Raises on
    any connection or protocol error.
    """
    message = f"[SendEmail_KTTT]{FACTORY}|{subject}|{body}"
    with socket.create_connection(
        (SERVER_IP, SERVER_PORT), timeout=SOCKET_TIMEOUT
    ) as sock:
        sock.sendall(message.encode("utf-8"))
        response = sock.recv(1024).decode("utf-8")
    return message, response


def send_results_email_async(subject: str, body: str, log=None) -> None:
    """Send the email on a fire-and-forget daemon thread.

    ``log`` (optional callable taking one string) reports success and
    failure. A dead network only costs one ``SOCKET_TIMEOUT`` wait inside
    the daemon thread plus a log line, never the watcher's processing loop.
    """

    def _worker() -> None:
        try:
            _, response = send_email(subject, body)
        except Exception as e:  # noqa: BLE001 — email must never break processing
            if log is not None:
                try:
                    log(f"[email] Lỗi kết nối: {e}")
                except Exception:
                    pass
            return
        if log is not None:
            try:
                log(f"[email] Đã gửi: {subject}")
                log(f"[email] Response: {response}")
            except Exception:
                pass

    threading.Thread(target=_worker, daemon=True, name="sendemail-kttt").start()


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _fmt_seconds(value) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return _DASH


def _event_row(index: int, event: dict, shaded: bool) -> str:
    behavior = str(event.get("behavior") or "").strip() or _DASH
    start = _fmt_seconds(event.get("start_time_sec"))
    end = _fmt_seconds(event.get("end_time_sec"))
    try:
        duration = f"{float(event['end_time_sec']) - float(event['start_time_sec']):g}"
    except (TypeError, ValueError, KeyError):
        duration = _DASH
    bg = "#f7f9fc" if shaded else "#ffffff"
    return (
        f'<tr style="background-color:{bg};">'
        f'<td style="{_CELL_CENTER}">{index}</td>'
        f'<td style="{_CELL_CENTER}">{_esc(event.get("event_id"))}</td>'
        f'<td style="{_CELL}">{_esc(behavior)}</td>'
        f'<td style="{_CELL_CENTER}">{start}</td>'
        f'<td style="{_CELL_CENTER}">{end}</td>'
        f'<td style="{_CELL_CENTER}">{duration}</td>'
        "</tr>"
    )


def build_results_html(
    video_name: str, output_dir: str, events, processed_at: str | None = None
) -> str:
    """Build a professional single-line HTML body for the results email.

    All dynamic values are HTML-escaped and every ``|`` is rewritten to
    ``/`` so the pipe-delimited socket protocol is never broken. Non-dict
    entries in ``events`` are ignored. When ``processed_at`` is omitted the
    current local time is used.
    """
    valid = [ev for ev in events if isinstance(ev, dict)]
    rows = "".join(
        _event_row(i, ev, shaded=(i % 2 == 0)) for i, ev in enumerate(valid, 1)
    )
    if processed_at is None:
        processed_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return (
        _TEMPLATE.replace("__FACTORY__", _esc(FACTORY))
        .replace("__VIDEO__", _esc(video_name))
        .replace("__TOTAL__", str(len(valid)))
        .replace("__ROWS__", rows)
        .replace("__TIME__", _esc(processed_at))
        .replace("__FOLDER__", _esc(output_dir))
        .replace("|", "/")
    )
