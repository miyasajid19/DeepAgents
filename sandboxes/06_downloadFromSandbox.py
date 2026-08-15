from pathlib import Path

from daytona import Daytona
from langchain_daytona import DaytonaSandbox


# ============================================================
# 1. Daytona client
# ============================================================

client = Daytona()


# ============================================================
# 2. Sandbox ID
# ============================================================

# Put the ID printed by your upload script here.
SANDBOX_ID = "65f8b262-ee5a-4770-a703-932c7e9e43e4"


# ============================================================
# 3. Get existing sandbox
# ============================================================

print(f"Getting sandbox: {SANDBOX_ID}")

sandbox = client.get(SANDBOX_ID)

print(f"Sandbox: {sandbox.id}")


# ============================================================
# 4. Start sandbox if it was stopped
# ============================================================

print("Starting sandbox...")

sandbox.start()

print("Sandbox started.")


# ============================================================
# 5. Daytona backend
# ============================================================

backend = DaytonaSandbox(
    sandbox=sandbox
)


# ============================================================
# 6. Files to download
# ============================================================

remote_files = [
    "/home/daytona/sajidmiya.txt",
    "/home/daytona/03_agentScoped.py",
    "/home/daytona/01_image.py",
]


# ============================================================
# 7. Download
# ============================================================

print("\nDownloading files...")

results = backend.download_files(remote_files)

# ============================================================
# 8. Save locally
# ============================================================

DOWNLOAD_DIR = Path("downloads").resolve()

DOWNLOAD_DIR.mkdir(exist_ok=True)


for result in results:

    if result.content is None:

        print(
            f"Failed to download "
            f"{result.path}: "
            f"{result.error}"
        )

        continue


    # --------------------------------------------------------
    # Extract filename
    # --------------------------------------------------------

    filename = Path(result.path).name


    local_path = (DOWNLOAD_DIR /filename)
    # --------------------------------------------------------
    # Write file
    # --------------------------------------------------------
    local_path.write_bytes(result.content)

    print(    f"Downloaded:    \n  Sandbox: {result.path}    \n  Local:   {local_path}")
# ============================================================
# 9. Stop sandbox
# ============================================================
print("\nStopping sandbox...")
sandbox.stop()
print("Sandbox stopped successfully.")