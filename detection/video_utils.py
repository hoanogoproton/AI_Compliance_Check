import cv2


def _even_size(size):
    w, h = size
    return (w // 2 * 2, h // 2 * 2)


def create_video_writer(path, fourcc_str, fps, size):
    even_size = _even_size(size)
    # mp4v first since avc1 often reports isOpened()=True but produces 0-byte files on Windows
    codes_to_try = ["mp4v", fourcc_str, "H264", "X264"]
    seen = set()
    deduped = []
    for c in codes_to_try:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    for code in deduped:
        fourcc = cv2.VideoWriter_fourcc(*code)
        writer = cv2.VideoWriter(str(path), fourcc, fps, even_size)
        if writer.isOpened():
            return writer
    raise RuntimeError(
        f"Could not open video writer at '{path}' — none of {deduped} codecs worked"
    )
