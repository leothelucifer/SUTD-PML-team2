import importlib
import pickle
import subprocess
import sys
import unittest
import warnings
from unittest import mock

from sepsis_warning import _legacy


class AliasTestValue:
    def __init__(self):
        self.x = 1


class LegacyImportTests(unittest.TestCase):
    def setUp(self):
        self.old_name = "sepsis_warning.cli_legacy"
        _legacy.LEGACY[self.old_name] = "sepsis_warning.cli"
        sys.modules.pop(self.old_name, None)

    def tearDown(self):
        _legacy.LEGACY.pop(self.old_name, None)
        sys.modules.pop(self.old_name, None)

    def test_alias_is_same_object_and_patch_affects_new_object(self):
        old = importlib.import_module(self.old_name)
        new = importlib.import_module("sepsis_warning.cli")
        self.assertIs(old, new)
        with mock.patch("sepsis_warning.cli_legacy.main", new=lambda: 123):
            self.assertEqual(new.main(), 123)

    def test_pickle_round_trip_uses_real_class(self):
        old = "test_module_legacy"
        new = f"{__name__}"
        _legacy.LEGACY[old] = new
        sys.modules.pop(old, None)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                importlib.import_module(old)
            blob = pickle.dumps(AliasTestValue())
            restored = pickle.loads(blob)
            self.assertIs(type(restored), AliasTestValue)
        finally:
            _legacy.LEGACY.pop(old, None)
            sys.modules.pop(old, None)

    def test_module_entry_point_resolves(self):
        env = dict(__import__("os").environ)
        env.pop("SEPSIS_WARNING_NO_LEGACY", None)
        with subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sepsis_warning, sepsis_warning._legacy as legacy; "
                f"legacy.LEGACY['{self.old_name}']='sepsis_warning.cli'; "
                f"module=__import__('importlib').import_module('{self.old_name}'); "
                "assert module.__name__ == 'sepsis_warning.cli'",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        ) as process:
            stdout, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0, stdout + stderr)

    def test_handler_resolution(self):
        old_name = "sepsis_warning.adapters.aws.lambda_app_legacy"
        _legacy.LEGACY[old_name] = "sepsis_warning.adapters.aws.lambda_app"
        sys.modules.pop(old_name, None)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                old = importlib.import_module(old_name)
            new = importlib.import_module("sepsis_warning.adapters.aws.lambda_app")
            self.assertIs(old, new)
            self.assertTrue(callable(old.handler))
        finally:
            _legacy.LEGACY.pop(old_name, None)
            sys.modules.pop(old_name, None)

    def test_deprecation_warning_is_raised(self):
        sys.modules.pop(self.old_name, None)
        with self.assertWarns(DeprecationWarning):
            importlib.import_module(self.old_name)


if __name__ == "__main__":
    unittest.main()
