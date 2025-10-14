import importlib
import subprocess
import sys

def ensure_dependency(package_name: str, git_url: str, optional: bool = False):
    """Check if a dependency is available; if not, optionally prompt to install."""
    try:
        importlib.import_module(package_name)
        print(f"✓ {package_name} available")
        return True
    except ImportError:
        print(f"✗ {package_name} not found")
        print(f"  Install manually with:\n  pip install git+{git_url}")
        if not optional:
            # Optionally auto-install (if you want automatic resolution)
            choice = input(f"Would you like to install {package_name}? [y/N]: ").strip().lower()
            if choice == "y":
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"git+{git_url}"])
        return False


def check_sam_dependencies(auto_install: bool = False):
    """Check for Segment Anything (SAM v1/v2) availability."""
    print("🔍 Checking SAM dependencies...\n")

    sam_v1 = ensure_dependency(
        package_name="segment_anything",
        git_url="https://github.com/facebookresearch/segment-anything.git",
        optional=not auto_install,
    )

    sam_v2 = ensure_dependency(
        package_name="sam2",
        git_url="https://github.com/facebookresearch/segment-anything-2.git",
        optional=True,
    )

    print("\n✅ Dependency check complete.\n")
    return sam_v1, sam_v2
