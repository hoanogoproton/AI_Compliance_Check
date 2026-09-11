"""
Email notifier for the folder watcher: sends video processing results to the
factory email service over a raw TCP socket.

Message format (single line, UTF-8, `|` as field delimiter):

    [SendEmail_KTTT]{factory}|{subject}|{body}

The results body is a single-line compact HTML document built by
``build_results_html``; ``|`` never appears in subject or body. The mail
service truncates requests that are too long (a large HTML body arrives
cut off mid-tag, e.g. right at ``<td style="padding:8px 10px;border``),
so the whole request is budgeted to stay under ``MAX_MESSAGE_BYTES``:
when not every event row fits, the first rows are embedded and a note
points to the output folder for the full list. Body text is Vietnamese
with diacritics, encoded as UTF-8 (the HTML declares ``charset="utf-8"``);
because accented characters take 2-3 bytes each, the budget is computed
on UTF-8 byte lengths.

In the results email the output folder is rendered as a clickable link:
absolute local paths become ``file:///`` URIs and ``http(s)://`` URLs are
kept as they are; anything else (relative paths) stays plain text. Note
that some mail clients (Outlook, Gmail) block ``file://`` links for
security reasons - publish the results on an internal file server and use
its URL as the output folder if the link must open for every recipient.

Edit the constants below to point at a different server or factory.
"""

from __future__ import annotations

import html
import socket
import threading
import urllib.parse
from datetime import datetime

SERVER_IP = "172.17.108.169"
SERVER_PORT = 1111
FACTORY = "PM6"
SOCKET_TIMEOUT = 10  # seconds

# The mail service cuts off long requests (the old rich-HTML body arrived
# truncated mid-table), so the whole request "envelope + subject + body" is
# budgeted to stay below MAX_MESSAGE_BYTES; _SUBJECT_RESERVE leaves room for
# the subject line.
MAX_MESSAGE_BYTES = 1900
_SUBJECT_RESERVE = 256

_DASH = "-"
_MORE_PARA = (
    '<p style="color:#c0392b;font-size:12px;">'
    "+ __N__ sự kiện khác - xem thư mục kết quả.</p>"
)

_TEMPLATE = (
    '<!DOCTYPE html><html><head><meta charset="utf-8"></head>'
    '<body style="padding:10px;font-family:Arial,Helvetica,sans-serif;'
    'font-size:13px;">'
    '<div style="background:#0b4f9e;color:#fff;padding:10px 12px;'
    'font-weight:bold;">BÁO CÁO KẾT QUẢ XỬ LÝ VIDEO'
    " - Nhà máy __FACTORY__</div>"
    '<p style="margin:10px 0;">Video <b>__VIDEO__</b>: ghi nhận '
    '<b style="color:#c0392b;">__TOTAL__ sự kiện</b>.</p>'
    '<table cellpadding="4" cellspacing="0" border="1"'
    ' style="border-collapse:collapse;">'
    '<tr bgcolor="#0b4f9e" style="color:#fff;">'
    "<th>Mã sự kiện</th><th>Hành vi</th>"
    "<th>Bắt đầu (s)</th><th>Kết thúc (s)</th></tr>"
    "__ROWS__"
    "</table>"
    "__MORE_PARA__"
    "<p>Thời gian xử lý: <b>__TIME__</b><br>"
    "Thư mục kết quả: __FOLDER_LINK__</p>"
    '<p style="color:#999;font-size:11px;">Email tự động - không trả lời.'
    " - Nhà máy __FACTORY__</p>"
    "</body></html>"
)


def send_email(subject: str, body: str) -> tuple[str, str]:
    """Send one email request and return ``(message, response)``.

    Blocking call: it can wait up to ``SOCKET_TIMEOUT`` seconds. Raises on
    connection errors. The service does not always reply before the
    timeout; when no response arrives in time the returned response is
    ``""`` even though the request itself was delivered.
    """
    message = f"[SendEmail_KTTT]{FACTORY}|{subject}|{body}"
    with socket.create_connection(
        (SERVER_IP, SERVER_PORT), timeout=SOCKET_TIMEOUT
    ) as sock:
        sock.sendall(message.encode("utf-8"))
        try:
            response = sock.recv(1024).decode("utf-8")
        except OSError:
            response = ""  # request delivered; no reply within the timeout
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
                if response:
                    log(f"[email] Response: {response}")
            except Exception:
                pass

    threading.Thread(target=_worker, daemon=True, name="sendemail-kttt").start()


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _file_uri(path_text: str) -> str | None:
    """Best-effort ``file:///`` URI for a local path, or ``None``.

    Absolute Windows drive paths (``D:/out``), UNC shares
    (``//server/share``) and POSIX absolute paths are supported; spaces and
    non-ASCII characters are percent-encoded. Relative paths return ``None``
    so the email keeps plain text instead of a broken link.
    """
    text = str(path_text).strip()
    if not text:
        return None
    if (
        len(text) >= 3
        and text[1] == ":"
        and text[2] in "/\\"
        and text[0].isascii()
        and text[0].isalpha()
    ):
        # Windows drive path -> file:///D:/out
        return "file:///" + urllib.parse.quote(text.replace("\\", "/"), safe="/:")
    if text.startswith("\\\\"):  # UNC share -> file://server/share
        return "file://" + urllib.parse.quote(text[2:].replace("\\", "/"), safe="/:")
    if text.startswith("/"):  # POSIX absolute path -> file:///home/user/out
        return "file://" + urllib.parse.quote(text, safe="/:")
    return None


def _folder_anchor(output_dir: str) -> str:
    """Render the output folder as a clickable link when possible.

    ``http(s)://`` URLs are linked as-is and absolute local paths become
    ``file:///`` links; anything else falls back to plain escaped text.
    """
    text = _esc(output_dir)
    stripped = str(output_dir).strip()
    if stripped.lower().startswith(("http://", "https://")):
        return f'<a href="{_esc(stripped)}">{text}</a>'
    uri = _file_uri(output_dir)
    if uri is None:
        return text
    return f'<a href="{_esc(uri)}">{text}</a>'


def _fmt_seconds(value) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return _DASH


def _event_row(event: dict) -> str:
    behavior = str(event.get("behavior") or "").strip() or _DASH
    start = _fmt_seconds(event.get("start_time_sec"))
    end = _fmt_seconds(event.get("end_time_sec"))
    return (
        "<tr>"
        f"<td>{_esc(event.get('event_id'))}</td>"
        f"<td>{_esc(behavior)}</td>"
        f"<td>{start}</td>"
        f"<td>{end}</td>"
        "</tr>"
    )


def build_results_html(
    video_name: str, output_dir: str, events, processed_at: str | None = None
) -> str:
    """Build a compact single-line HTML body for the results email.

    All dynamic values are HTML-escaped and every ``|`` is rewritten to
    ``/`` so the pipe-delimited socket protocol is never broken. Non-dict
    entries in ``events`` are ignored. When ``processed_at`` is omitted the
    current local time is used.

    The result is sized so the full socket request
    ``[SendEmail_KTTT]{factory}|{subject}|{body}`` stays under
    ``MAX_MESSAGE_BYTES`` (the mail service truncates longer requests).
    When there is not enough room for every event row, only the first rows
    are embedded and a note states how many more events are listed in the
    output folder.

    The output folder is rendered as a clickable link when ``output_dir``
    is an absolute local path (``file:///`` URI) or an ``http(s)://`` URL;
    otherwise it stays plain text. The link is part of the fixed overhead,
    so the row budget automatically accounts for its bytes.
    """
    valid = [ev for ev in events if isinstance(ev, dict)]
    if processed_at is None:
        processed_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    head, tail = _TEMPLATE.split("__ROWS__")
    head = (
        head.replace("__FACTORY__", _esc(FACTORY))
        .replace("__VIDEO__", _esc(video_name))
        .replace("__TOTAL__", str(len(valid)))
    )
    tail = (
        tail.replace("__TIME__", _esc(processed_at))
        .replace("__FOLDER_LINK__", _folder_anchor(output_dir))
        .replace("__FACTORY__", _esc(FACTORY))
    )
    budget = (
        MAX_MESSAGE_BYTES
        - len(f"[SendEmail_KTTT]{FACTORY}|".encode("utf-8"))
        - _SUBJECT_RESERVE
    )
    used = len(head.encode("utf-8")) + len(tail.encode("utf-8"))
    # Reserve room for the "+N sự kiện khác" note up-front so the note can
    # always be appended whenever rows have to be dropped.
    row_budget = budget - len(
        _MORE_PARA.replace("__N__", "99999").encode("utf-8")
    )
    rows: list[str] = []
    for event in valid:
        row = _event_row(event)
        size = len(row.encode("utf-8"))
        if rows and used + size > row_budget:
            break
        rows.append(row)
        used += size
    hidden = len(valid) - len(rows)
    if hidden:
        more = _MORE_PARA.replace("__N__", str(hidden))
        if used + len(more.encode("utf-8")) <= budget:
            tail = tail.replace("__MORE_PARA__", more)
    return (
        (head + "".join(rows) + tail)
        .replace("__MORE_PARA__", "")
        .replace("|", "/")
    )
