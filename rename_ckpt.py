import argparse
import glob
import os
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description="Find latest model checkpoint and rename/move it.")
parser.add_argument("--pattern", default="**/*.{pth,pt,ckpt,safetensors,h5}", help="glob pattern (supports braces)")
parser.add_argument("--dest", default="best_model.pth", help="destination filename (in cwd or specify subdir)")
parser.add_argument("--move", action="store_true", help="move instead of copy")
parser.add_argument("--dry", action="store_true", help="dry run, only print actions")
args = parser.parse_args()

# expand brace-style patterns for glob if needed
patterns = []
pat = args.pattern
if "{" in pat and "}" in pat:
    pre, brace = pat.split("{",1)
    inner, post = brace.split("}",1)
    for ext in inner.split(","):
        patterns.append(pre + ext + post)
else:
    patterns = [pat]

matches = []
for p in patterns:
    matches += glob.glob(p, recursive=True)

matches = [m for m in matches if os.path.isfile(m)]
if not matches:
    print("No checkpoint files found with pattern:", args.pattern)
    raise SystemExit(1)

# choose latest by mtime
matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
src = matches[0]
dest = os.path.abspath(args.dest)

print("Latest checkpoint:", src)
print("Destination:", dest)
if args.dry:
    print("Dry run, no changes made.")
    raise SystemExit(0)

os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
if args.move:
    shutil.move(src, dest)
    print("Moved.")
else:
    shutil.copy2(src, dest)
    print("Copied.")