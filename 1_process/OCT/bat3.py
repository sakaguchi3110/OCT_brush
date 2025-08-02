import subprocess

scripts = [
    "1_4_1_process_oct_skin-surface_tracking_correction.py",
    "1_5_1_process_oct_phaseFFT.py"
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)
