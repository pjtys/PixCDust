from pathlib import Path

import pytest

from pixcdust.tests.init_tests import JsonTestsSettings, init_hydroweb_env

MOCK_DATA_DIR = Path(__file__).parent / "mock_data"
SAMPLE_NC = MOCK_DATA_DIR / "swot_pixc.nc"


def pytest_addoption(parser):
    parser.addoption(
        "--dl", action="store_true", default=False, help="run dowloaders tests"
    )
    parser.addoption(
        "--realdata",
        action="store_true",
        default=False,
        help="run tests on real SWOT data (requires API key)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "downloader: mark test as testing downloads")
    config.addinivalue_line(
        "markers", "realdata: mark test as requireing real SWOT data"
    )


def pytest_collection_modifyitems(config, items):
    skip_dl = pytest.mark.skip(reason="need --dl option to run")
    skip_realdata = pytest.mark.skip(
        reason="need --realdata option and a configured input_folder"
    )
    for item in items:
        if "downloader" in item.keywords and not config.getoption("--dl"):
            item.add_marker(skip_dl)
        if "realdata" in item.keywords and not config.getoption("--realdata"):
            item.add_marker(skip_realdata)


@pytest.fixture(scope="session")
def tests_settings() -> JsonTestsSettings:
    return JsonTestsSettings()


@pytest.fixture(scope="session")
def sample_nc() -> Path:
    assert SAMPLE_NC.is_file()
    print(SAMPLE_NC)
    return SAMPLE_NC


@pytest.fixture(scope="session")
def input_folder(tests_settings) -> Path:
    try:
        return tests_settings.input_folder
    except KeyError:
        return MOCK_DATA_DIR


@pytest.fixture(scope="session")
def input_files(input_folder) -> list[Path]:
    return list(input_folder.glob("**/*nc"))


@pytest.fixture(scope="session")
def first_file(input_folder, sample_nc) -> Path:
    return next(iter(input_folder.glob("**/*_20240803T*nc")), sample_nc)


@pytest.fixture(scope="session")
def tmp_folder(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("pixcdust-test")


@pytest.fixture()
def hydroweb_env(tests_settings) -> None:
    init_hydroweb_env(tests_settings)
