#!/bin/bash
nohup python -u run.py \
    --test_file ./data/laptop.jsonl \
    --api_key YOUR_GEMINI_API_KEY \
    --headless \
    --max_iter 15 \
    --max_attached_imgs 3 \
    --temperature 1 \
    --fix_box_color \
    --seed 42 > test_tasks.log &
