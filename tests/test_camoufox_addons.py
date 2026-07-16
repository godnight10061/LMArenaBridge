import os
import unittest

from camoufox.addons import DefaultAddons

from src import main


class TestCamoufoxAddons(unittest.TestCase):
    def test_all_default_addons_are_excluded(self):
        options = main._camoufox_launch_options(headless=True)

        self.assertTrue(options["headless"])
        self.assertTrue(options["main_world_eval"])
        self.assertEqual(set(options["exclude_addons"]), set(DefaultAddons))

    def test_system_chromium_is_available_for_fallback(self):
        executable = main._find_chromium_executable()

        self.assertIsNotNone(executable)
        self.assertTrue(os.path.isfile(executable))
