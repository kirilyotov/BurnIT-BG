import importlib


def test_import_utils_package() -> None:
	assert importlib.import_module("utils") is not None


def test_import_utils_modules() -> None:
	assert importlib.import_module("utils.argparser") is not None
	assert importlib.import_module("utils.plots") is not None
