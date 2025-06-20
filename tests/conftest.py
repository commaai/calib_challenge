import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_test_data(temp_dir):
    """Create mock test data files."""
    data = {}
    for i in range(5):
        arr = np.random.rand(10, 10)
        np.savetxt(temp_dir / f"{i}.txt", arr)
        data[i] = arr
    return data


@pytest.fixture
def mock_labeled_data(temp_dir):
    """Create mock labeled data files."""
    labeled_dir = temp_dir / "labeled"
    labeled_dir.mkdir()
    
    data = {}
    for i in range(5):
        arr = np.random.rand(10, 10)
        np.savetxt(labeled_dir / f"{i}.txt", arr)
        data[i] = arr
    return labeled_dir, data


@pytest.fixture
def sample_numpy_array():
    """Provide a sample numpy array for testing."""
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary."""
    return {
        "test_dir": "tests/data",
        "gt_dir": "labeled/",
        "output_format": "txt",
        "precision": 2,
    }


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_output)
    return captured_output


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset imported modules between tests to avoid state pollution."""
    import sys
    modules_to_reset = [m for m in sys.modules if m.startswith('eval')]
    for module in modules_to_reset:
        del sys.modules[module]


@pytest.fixture
def mock_sys_argv(monkeypatch):
    """Mock sys.argv for testing command line arguments."""
    def _mock_argv(args):
        monkeypatch.setattr('sys.argv', args)
    return _mock_argv