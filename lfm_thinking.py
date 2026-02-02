"""
LFM-2.5 Inference Script (Text & Vision)
========================================
Written by Jonathan M Rothberg

Runs LiquidAI's LFM models locally:
- LFM2.5-1.2B-Thinking: Text-only reasoning model
- LFM2.5-VL-1.6B: Vision-Language model with image/video support

USAGE: python lfm_thinking.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from PIL import Image
from datetime import datetime
import torch
import os
import cv2
import time
import tkinter as tk
from tkinter import filedialog

# Hide tkinter root window
tk.Tk().withdraw()

# Blackwell GPU optimizations
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Local model paths
TEXT_MODEL_PATH = "/home/jonathan/Models_Transformer/LFM2.5-1.2B-Thinking"
VL_MODEL_PATH = "/home/jonathan/Models_Transformer/LFM2.5-VL-1.6B"

# ============================================================================
# Model Selection
# ============================================================================
print("=" * 50)
print("LFM-2.5 Model Selection")
print("=" * 50)
print("1. LFM2.5-1.2B-Thinking (Text-only, reasoning)")
print("2. LFM2.5-VL-1.6B (Vision-Language, images + video)")
print("=" * 50)

while True:
    choice = input("Select model (1 or 2): ").strip()
    if choice in {"1", "2"}:
        break
    print("Please enter 1 or 2")

use_vl_model = (choice == "2")

# ============================================================================
# Load Selected Model
# ============================================================================
if use_vl_model:
    print("\nLoading LFM2.5-VL-1.6B (Vision-Language)...")
    model = AutoModelForImageTextToText.from_pretrained(
        VL_MODEL_PATH,
        device_map="auto",
        dtype="bfloat16",
        trust_remote_code=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(VL_MODEL_PATH, trust_remote_code=True, local_files_only=True)
    print("\nLFM2.5-VL-1.6B Interactive Chat")
    print("=" * 50)
    print("Commands: i=image, v=video, n=text-only, quit=exit")
    print("=" * 50 + "\n")
else:
    print("\nLoading LFM2.5-1.2B-Thinking (Text-only)...")
    model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL_PATH,
        device_map="auto",
        dtype="bfloat16",
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH, trust_remote_code=True, local_files_only=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print("\nLFM2.5-1.2B-Thinking Interactive Chat")
    print("=" * 50)
    print("Type 'quit' to exit.")
    print("=" * 50 + "\n")

# ============================================================================
# Main Chat Loop
# ============================================================================
while True:
    try:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if not user_input:
            print("Please enter a prompt...")
            continue

        if use_vl_model:
            media_choice = input("Media type? (i=image, v=video, n=none): ").strip().lower()

            # ----------------------------------------------------------------
            # VIDEO MODE: Extract frames at interval and describe each
            # ----------------------------------------------------------------
            if media_choice in {"v", "video"}:
                print("Opening file dialog for video...")
                video_path = filedialog.askopenfilename(
                    title="Select a video",
                    filetypes=[
                        ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                        ("MP4", "*.mp4"),
                        ("AVI", "*.avi"),
                        ("All files", "*.*"),
                    ]
                )

                if not video_path:
                    print("No video selected.")
                    continue

                print(f"Loading video: {video_path}")
                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    print("Error: Could not open video.")
                    continue

                # Get video metadata
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0

                print(f"Video: {fps:.1f} FPS, {total_frames} frames, {duration:.1f}s duration")

                # Set frame sampling interval
                interval_input = input("Analyze every N seconds (default=2): ").strip()
                interval_seconds = float(interval_input) if interval_input else 2.0
                frame_interval = int(fps * interval_seconds)

                print(f"Sampling every {interval_seconds}s ({frame_interval} frames)")
                print("=" * 50)
                print("VIDEO SCENE DESCRIPTIONS:")
                print("=" * 50)

                # Store results for optional save
                video_results = []
                video_name = os.path.basename(video_path)
                frame_count = 0
                scene_count = 0
                start_time = time.time()

                # Process video frame by frame, analyze at intervals
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        scene_count += 1
                        timestamp = frame_count / fps

                        # Convert OpenCV BGR to RGB PIL Image
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)

                        # Build conversation with image
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": pil_image},
                                    {"type": "text", "text": user_input},
                                ],
                            },
                        ]

                        # Generate description
                        inputs = processor.apply_chat_template(
                            conversation,
                            add_generation_prompt=True,
                            return_tensors="pt",
                            return_dict=True,
                            tokenize=True,
                        ).to(model.device)

                        outputs = model.generate(**inputs, max_new_tokens=128)
                        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

                        # Extract assistant response
                        if "assistant" in response.lower():
                            response = response.split("assistant")[-1].strip()

                        print(f"\n[{timestamp:.1f}s] Scene {scene_count}:")
                        print(f"  {response}")

                        # Store for save
                        video_results.append({
                            "timestamp": timestamp,
                            "scene": scene_count,
                            "description": response
                        })

                    frame_count += 1

                cap.release()
                elapsed = time.time() - start_time

                print("\n" + "=" * 50)
                print(f"Analysis complete: {scene_count} scenes in {elapsed:.1f}s")
                print(f"Average: {elapsed/scene_count:.2f}s per scene")
                print("=" * 50)

                # Offer to save results
                save_choice = input("Save results? (y/n): ").strip().lower()
                if save_choice in {"y", "yes"}:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_basename = os.path.splitext(video_name)[0]
                    output_filename = f"{video_basename}_analysis_{timestamp_str}.txt"

                    with open(output_filename, "w") as f:
                        f.write(f"Video Analysis: {video_name}\n")
                        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Prompt: {user_input}\n")
                        f.write(f"Interval: {interval_seconds}s\n")
                        f.write(f"Total scenes: {scene_count}\n")
                        f.write(f"Processing time: {elapsed:.1f}s\n")
                        f.write("=" * 50 + "\n\n")

                        for result in video_results:
                            f.write(f"[{result['timestamp']:.1f}s] Scene {result['scene']}:\n")
                            f.write(f"  {result['description']}\n\n")

                    print(f"Saved to: {output_filename}")

            # ----------------------------------------------------------------
            # IMAGE MODE: Single image analysis
            # ----------------------------------------------------------------
            elif media_choice in {"i", "image", "y", "yes"}:
                print("Opening file dialog for image...")
                image_path = filedialog.askopenfilename(
                    title="Select an image",
                    filetypes=[
                        ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                        ("PNG", "*.png"),
                        ("JPEG", "*.jpg *.jpeg"),
                        ("All files", "*.*"),
                    ]
                )

                if image_path:
                    try:
                        print(f"Loading image: {image_path}")
                        image = load_image(image_path)
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": user_input},
                                ],
                            },
                        ]
                    except Exception as img_err:
                        print(f"Error loading image: {img_err}")
                        conversation = [{"role": "user", "content": [{"type": "text", "text": user_input}]}]
                else:
                    print("No image selected.")
                    conversation = [{"role": "user", "content": [{"type": "text", "text": user_input}]}]

                print("Assistant: ", end="", flush=True)

                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                ).to(model.device)

                outputs = model.generate(**inputs, max_new_tokens=512)
                response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

                if "assistant" in response.lower():
                    response = response.split("assistant")[-1].strip()

                print(response)
                print("\n" + "=" * 50)

            # ----------------------------------------------------------------
            # TEXT ONLY MODE
            # ----------------------------------------------------------------
            else:
                conversation = [{"role": "user", "content": [{"type": "text", "text": user_input}]}]

                print("Assistant: ", end="", flush=True)

                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                ).to(model.device)

                outputs = model.generate(**inputs, max_new_tokens=512)
                response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

                if "assistant" in response.lower():
                    response = response.split("assistant")[-1].strip()

                print(response)
                print("\n" + "=" * 50)

        # ====================================================================
        # Text-Only Model Mode
        # ====================================================================
        else:
            print("Assistant: ", end="", flush=True)

            messages = [{"role": "user", "content": user_input}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
            )

            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.1,
                top_k=50,
                top_p=0.1,
                repetition_penalty=1.05,
                max_new_tokens=512,
                streamer=streamer,
            )

            print("\n" + "=" * 50)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Type 'quit' to exit or continue chatting.")
        continue
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {str(e)}")
        print("Try again or type 'quit' to exit.")
        continue
