import subprocess

scripts = [
    "1_3_0_process_oct_phase-change.py",
    "1_4_0_process_oct_skin-surface_tracking.py"
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)
