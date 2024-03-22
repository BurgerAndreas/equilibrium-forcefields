# Real programmers would use environment variables but I am not a real programmer
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOWNLOAD_DIR = f"{ROOT_DIR}/data"
OUTPUT_DIR = f"{ROOT_DIR}/output"
CHECKPOINT_DIR = f"{ROOT_DIR}/output/checkpoint"