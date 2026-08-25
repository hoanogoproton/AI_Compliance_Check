import subprocess
from pathlib import Path

input_video = r"D:\Video\CA170-FB-PC-No5-20260824-114244.mp4"
output_dir = Path("frames_20260824-114244")
output_dir.mkdir(exist_ok=True)

output_pattern = output_dir / "frame_%06d.jpg"

command = [
    "ffmpeg",
    "-i", input_video,

    # Lấy các frame: 0, 5, 10, 15, ...
    "-vf", r"select='not(mod(n\,20))'",

    # Thay cho -vsync vfr trong FFmpeg 9
    "-fps_mode", "vfr",

    # Chất lượng JPEG, số càng nhỏ chất lượng càng cao
    "-q:v", "2",

    "-y",
    str(output_pattern)
]

try:
    subprocess.run(command, check=True)
    print(f"Đã trích xuất ảnh vào: {output_dir.resolve()}")

except FileNotFoundError:
    print("Không tìm thấy FFmpeg trong PATH.")

except subprocess.CalledProcessError as error:
    print("FFmpeg xử lý thất bại.")
    print(error)