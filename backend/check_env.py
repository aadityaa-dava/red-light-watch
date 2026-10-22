"""Environment check: run inside the Docker container on the Jetson."""
import platform
import sys


def check(name, fn):
    try:
        print(f"[ OK ] {name}: {fn()}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        return False


def torch_info():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available (use --runtime nvidia)")
    return f"torch {torch.__version__}, GPU = {torch.cuda.get_device_name(0)}"


def main():
    results = [
        check("Python", lambda: platform.python_version()),
        check("PyTorch + CUDA", torch_info),
        check("OpenCV", lambda: __import__("cv2").__version__),
        check("Ultralytics", lambda: __import__("ultralytics").__version__),
        check("supervision", lambda: __import__("supervision").__version__),
    ]
    print("\nAll checks passed." if all(results) else "\nSome checks failed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
