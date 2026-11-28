"""Research scaffold for the PhysioNet 2019 early-warning task."""

import importlib
import importlib.util
import os
import sys
import warnings
from importlib.abc import Loader, MetaPathFinder

from ._legacy import LEGACY


class _LegacyLoader(Loader):
    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        warnings.warn(
            f"{self.old_name} is deprecated; import {self.new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        new_module = importlib.import_module(self.new_name)
        sys.modules[self.old_name] = new_module
        module.__dict__.update(new_module.__dict__)
        module.__spec__ = importlib.util.find_spec(self.new_name)
        module.__package__ = new_module.__package__


class _LegacyFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if os.environ.get("SEPSIS_WARNING_NO_LEGACY") or fullname not in LEGACY:
            return None
        new_name = LEGACY[fullname]
        loader = _LegacyLoader(fullname, new_name)
        return importlib.util.spec_from_loader(fullname, loader)


if not os.environ.get("SEPSIS_WARNING_NO_LEGACY"):
    _finder = _LegacyFinder()
    if not any(isinstance(finder, _LegacyFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _finder)
