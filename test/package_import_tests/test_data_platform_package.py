import importlib


def test_import_data_platform_package() -> None:
	assert importlib.import_module("data_platform") is not None


def test_import_data_platform_common_modules() -> None:
	assert importlib.import_module("data_platform.common") is not None
	assert importlib.import_module("data_platform.common.config") is not None
	assert importlib.import_module("data_platform.common.env") is not None
	assert importlib.import_module("data_platform.common.exceptions") is not None


def test_import_data_platform_dataset_modules() -> None:
	assert importlib.import_module("data_platform.datasets") is not None
	assert importlib.import_module("data_platform.datasets.cli") is not None
	assert importlib.import_module("data_platform.datasets.minio") is not None
	assert importlib.import_module("data_platform.datasets.service") is not None


def test_import_data_platform_storage_modules() -> None:
	assert importlib.import_module("data_platform.storage") is not None
	assert importlib.import_module("data_platform.storage.base") is not None
	assert importlib.import_module("data_platform.storage.hugging_face") is not None
	assert importlib.import_module("data_platform.storage.local") is not None
	assert importlib.import_module("data_platform.storage.minio") is not None


def test_import_data_platform_tracking_modules() -> None:
	assert importlib.import_module("data_platform.tracking") is not None
	assert importlib.import_module("data_platform.tracking.mlflow") is not None
