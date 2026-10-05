import sys
import os
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.visionqc.api.main import app

def run_tests():
    with TestClient(app) as client:
        print("Testing /health endpoint...")
        response = client.get("/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        print("Health check passed.\n")

        print("Testing /predict endpoint with a valid image...")
        # Find a test image
        images_dir = _PROJECT_ROOT / "data" / "prepared" / "test" / "images"
        test_images = list(images_dir.glob("*.jpg"))
        if not test_images:
            print("No test images found. Skipping valid image test.")
        else:
            test_image_path = test_images[0]
            print(f"Using test image: {test_image_path}")
            with open(test_image_path, "rb") as f:
                response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})
            print(f"Status Code: {response.status_code}")
            response_data = response.json()
            print(f"Predictions: {len(response_data.get('predictions', []))} boxes found.")
            assert response.status_code == 200
            assert "predictions" in response_data
            print("Valid prediction check passed.\n")

        print("Testing /predict endpoint with invalid input (text file)...")
        invalid_content = b"This is not an image."
        response = client.post("/predict", files={"file": ("test.txt", invalid_content, "text/plain")})
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 400
        print("Invalid input check passed.\n")

if __name__ == "__main__":
    run_tests()
    print("All tests completed successfully!")
