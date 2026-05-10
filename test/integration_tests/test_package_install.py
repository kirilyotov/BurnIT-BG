"""Integration tests for package installation (local and from GitHub)."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class TestPackageInstall:
    """Test suite for package installation methods."""

    @staticmethod
    def run_pip_command(command: list[str], target_dir: Path | None = None) -> subprocess.CompletedProcess:
        """Run a pip command with optional target directory for isolation."""
        cmd = [sys.executable, "-m", "pip", "install"] + command
        
        # Add --target for isolation if provided
        if target_dir:
            target_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--target", str(target_dir)])
        
        return subprocess.run(cmd, capture_output=True, text=True)

    @staticmethod
    def verify_imports(target_dir: Path | None = None) -> bool:
        """Verify that package imports work."""
        import_code = "import data_platform; import utils; print('OK')"
        
        if target_dir:
            # Add target to Python path for import check
            env = {**subprocess.os.environ, "PYTHONPATH": str(target_dir)}
            result = subprocess.run(
                [sys.executable, "-c", import_code],
                capture_output=True,
                text=True,
                env=env
            )
        else:
            result = subprocess.run(
                [sys.executable, "-c", import_code],
                capture_output=True,
                text=True
            )
        
        return result.returncode == 0 and "OK" in result.stdout

    def test_local_install(self):
        """Test installing package locally with `pip install .`"""
        repo_root = Path(__file__).parent.parent.parent
        target_dir = Path(tempfile.mkdtemp(prefix="burnit_install_"))

        try:
            # Install package locally
            result = self.run_pip_command([str(repo_root)], target_dir)
            assert result.returncode == 0, f"Local install failed:\n{result.stderr}"

            # Verify imports work
            assert self.verify_imports(target_dir), "Package imports failed after local install"

        finally:
            # Cleanup
            shutil.rmtree(target_dir, ignore_errors=True)

    def test_local_editable_install(self):
        """Test installing package in editable/development mode with `pip install -e .`"""
        repo_root = Path(__file__).parent.parent.parent
        target_dir = Path(tempfile.mkdtemp(prefix="burnit_editable_"))

        try:
            # Install package in editable mode using --target for isolation
            # (full editable mode requires venv, --target provides reasonable isolation)
            cmd = [sys.executable, "-m", "pip", "install", "-e", str(repo_root), "--target", str(target_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"Editable install failed:\n{result.stderr}"

            # Verify imports work
            assert self.verify_imports(target_dir), "Package imports failed after editable install"

        finally:
            # Cleanup
            shutil.rmtree(target_dir, ignore_errors=True)

    def test_github_install(self):
        """Test installing package from GitHub with git+https URL."""
        target_dir = Path(tempfile.mkdtemp(prefix="burnit_github_"))

        try:
            # Install package from GitHub
            github_url = "git+https://github.com/kirilyotov/BurnIT-BG.git"
            result = self.run_pip_command([github_url], target_dir)
            assert result.returncode == 0, f"GitHub install failed:\n{result.stderr}"

            # Verify imports work
            assert self.verify_imports(target_dir), "Package imports failed after GitHub install"

        finally:
            # Cleanup
            shutil.rmtree(target_dir, ignore_errors=True)

    def test_github_install_with_branch(self):
        """Test installing package from a specific GitHub branch."""
        target_dir = Path(tempfile.mkdtemp(prefix="burnit_branch_"))

        try:
            # Install package from GitHub master branch
            github_url = "git+https://github.com/kirilyotov/BurnIT-BG.git@master"
            result = self.run_pip_command([github_url], target_dir)
            assert result.returncode == 0, f"GitHub branch install failed:\n{result.stderr}"

            # Verify imports work
            assert self.verify_imports(target_dir), "Package imports failed after GitHub branch install"

        finally:
            # Cleanup
            shutil.rmtree(target_dir, ignore_errors=True)

    def test_package_metadata(self):
        """Test that installed package has correct metadata."""
        target_dir = Path(tempfile.mkdtemp(prefix="burnit_meta_"))

        try:
            # Install package locally
            repo_root = Path(__file__).parent.parent.parent
            result = self.run_pip_command([str(repo_root)], target_dir)
            assert result.returncode == 0, f"Install failed:\n{result.stderr}"

            # Check version metadata with proper Python multiline code
            check_code = """import importlib.metadata
try:
    pkg = importlib.metadata.distribution('burnit_bg')
    print(f'Package: {pkg.name}, Version: {pkg.version}')
except:
    print('NOT FOUND')
"""
            env = {**subprocess.os.environ, "PYTHONPATH": str(target_dir)}
            result = subprocess.run(
                [sys.executable, "-c", check_code],
                capture_output=True,
                text=True,
                env=env
            )
            assert result.returncode == 0, f"Version check failed:\n{result.stderr}"
            assert ("Package: burnit_bg" in result.stdout or "Package: burnit-bg" in result.stdout), \
                f"Package metadata not found:\n{result.stdout}"
            assert "Version: 0.1.0" in result.stdout, \
                f"Expected version 0.1.0:\n{result.stdout}"

        finally:
            # Cleanup
            shutil.rmtree(target_dir, ignore_errors=True)
