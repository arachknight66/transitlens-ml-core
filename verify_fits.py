import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api.app import app

client = TestClient(app)

fits_path = r"C:\Users\arach\Documents\Projects\Transitlens\transitlens-data-pipeline\real_tess\cache\TIC261136679_sector095.fits"

if not os.path.exists(fits_path):
    print(f"FITS file not found at {fits_path}")
    sys.exit(1)

with open(fits_path, "rb") as f:
    response = client.post(
        "/analyze/file",
        files={"file": ("TIC261136679_sector095.fits", f, "application/octet-stream")},
        data={"target_id": "TIC 261136679", "metadata": "{}"}
    )

print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Detected: {data.get('candidate_detected')}")
    print(f"Class: {data.get('predicted_class')}")
    print(f"Explanation: {data.get('explanation')}")
else:
    print(response.text)
