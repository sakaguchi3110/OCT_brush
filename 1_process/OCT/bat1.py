import subprocess

scripts = [
    "1_0_0_process_oct_structural_images.py",
    "1_1_1_process_oct_video_vibrations.py"
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)
