from pathlib import Path
from daytona import Daytona
from langchain_daytona import DaytonaSandbox
client = Daytona()
sandbox = client.create()
backend = DaytonaSandbox(sandbox=sandbox)
print(f"Sandbox created: {sandbox.id}")
import time
time.sleep(10)
files = [
    Path(r"E:\Deep Agents\sajidmiya.txt"),
    Path(r"E:\Deep Agents\sandboxes\03_agentScoped.py"),
    Path(r"E:\Deep Agents\Multi-modal-inputs\01_image.py"),
]
uploads = []
for local_file in files:
    if not local_file.exists():
        print(f"Skipping: {local_file}")
        continue
    destination = (f"/home/daytona/{local_file.name}")
    uploads.append((destination,local_file.read_bytes(),))
backend.upload_files(uploads)
print("\nUploaded files:")
for destination, _ in uploads:
    print(destination)
time.sleep(10)
sandbox.delete()
# client.delete(sandbox,wait=True)