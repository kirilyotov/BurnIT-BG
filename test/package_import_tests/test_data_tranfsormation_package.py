import importlib


def test_import_data_transformation_package() -> None:
	assert importlib.import_module("data_transformation") is not None
