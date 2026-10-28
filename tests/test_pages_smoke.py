"""
Smoke tests for CSD Energy Intelligence Streamlit application.
Validates that Module 1 (Overview), Module 2 (Energy Analysis placeholder),
and Module 3 (Usage Patterns placeholder) run without exceptions.
"""

from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

BASE_DIR = Path(__file__).resolve().parent.parent


class TestStreamlitPages(unittest.TestCase):

    def test_overview_page(self):
        """Validates that the Overview Dashboard (app.py) runs completely without exception."""
        target = str(BASE_DIR / "app.py")
        at = AppTest.from_file(target).run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions in app.py: {at.exception}")

    def test_energy_analysis_page(self):
        """Validates that Module 2 (Energy Analysis) runs without exception and handles filters."""
        target = str(BASE_DIR / "pages" / "1_Energy_Analysis.py")
        at = AppTest.from_file(target).run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions in Energy Analysis: {at.exception}")

        # Test selecting 3rd Floor
        at.selectbox(key=None).select("3rd Floor").run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions on 3rd Floor filter: {at.exception}")

        # Test switching View By to Daily
        at.selectbox[1].select("Daily").run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions on Daily view: {at.exception}")

        # Test switching View By to Weekly
        at.selectbox[1].select("Weekly").run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions on Weekly view: {at.exception}")

    def test_usage_patterns_page(self):
        """Validates that Module 3 (Usage Patterns) runs without exception and handles filters."""
        target = str(BASE_DIR / "pages" / "2_Usage_Patterns.py")
        at = AppTest.from_file(target).run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions in Usage Patterns: {at.exception}")

        # Test selecting 3rd Floor
        at.selectbox[0].select("3rd Floor").run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions on 3rd Floor filter: {at.exception}")

        # Test selecting a specific monitored space (DSBD Lab)
        at.selectbox[1].select("DSBD Lab").run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions on DSBD Lab selection: {at.exception}")

        # Test modifying academic start hour
        at.selectbox[2].select(8).run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions on Academic Start Hour change: {at.exception}")

        # Test modifying academic end hour
        at.selectbox[3].select(18).run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions on Academic End Hour change: {at.exception}")

    def test_images_exist_and_readable(self):
        """Validates that supplied photographs and room assets exist in assets."""
        college_img = BASE_DIR / "assets" / "college_building.png"
        concept_img = BASE_DIR / "assets" / "energy_concept.jpg"
        feathered_img = BASE_DIR / "assets" / "college_building_feathered.png"
        self.assertTrue(college_img.exists(), "assets/college_building.png not found")
        self.assertTrue(concept_img.exists(), "assets/energy_concept.jpg not found")
        self.assertTrue(feathered_img.exists(), "assets/college_building_feathered.png not found")
        self.assertGreater(college_img.stat().st_size, 10000, "college_building.png is too small or corrupt")
        self.assertGreater(concept_img.stat().st_size, 10000, "energy_concept.jpg is too small or corrupt")

        # Verify all 8 gallery thumbnails exist
        gallery_rooms = [
            "dsbd_lab.jpg", "dbms_lab.jpg", "classroom_e501.jpg", "classroom_e502.jpg",
            "dsa_lab.jpg", "pbl_lab.jpg", "hod_cabin.jpg", "department_hall.jpg"
        ]
        for g_img in gallery_rooms:
            p = BASE_DIR / "assets" / "gallery_rooms" / g_img
            self.assertTrue(p.exists(), f"Gallery image missing: {g_img}")
            self.assertGreater(p.stat().st_size, 5000, f"Gallery image too small: {g_img}")

    def test_tariff_setting_default_and_modification(self):
        """Validates that tariff defaults to 11.00 and modifies cleanly via widget."""
        target = str(BASE_DIR / "app.py")
        at = AppTest.from_file(target).run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions in app.py: {at.exception}")
        self.assertAlmostEqual(at.session_state.get("tariff_rate", 0.0), 11.00, places=2)

        # Modify tariff widget to 14.50
        tariff_input = at.number_input(key="tariff_rate_widget")
        self.assertIsNotNone(tariff_input)
        tariff_input.set_value(14.50).run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions after changing tariff: {at.exception}")
        self.assertAlmostEqual(at.session_state.get("tariff_rate", 0.0), 14.50, places=2)

    def test_energy_analysis_tariff_integration(self):
        """Validates that Energy Analysis recalculates costs when tariff changes without errors."""
        target = str(BASE_DIR / "pages" / "1_Energy_Analysis.py")
        at = AppTest.from_file(target).run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions in Energy Analysis: {at.exception}")
        self.assertAlmostEqual(at.session_state.get("tariff_rate", 0.0), 11.00, places=2)

        # Modify tariff to 16.00
        tariff_input = at.number_input(key="tariff_rate_widget")
        self.assertIsNotNone(tariff_input)
        tariff_input.set_value(16.00).run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions in Energy Analysis on tariff change: {at.exception}")
        self.assertAlmostEqual(at.session_state.get("tariff_rate", 0.0), 16.00, places=2)

    def test_usage_patterns_tariff_integration(self):
        """Validates that Usage Patterns recalculates period costs when tariff changes without errors."""
        target = str(BASE_DIR / "pages" / "2_Usage_Patterns.py")
        at = AppTest.from_file(target).run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions in Usage Patterns: {at.exception}")
        self.assertAlmostEqual(at.session_state.get("tariff_rate", 0.0), 11.00, places=2)

        # Modify tariff to 12.50
        tariff_input = at.number_input(key="tariff_rate_widget")
        self.assertIsNotNone(tariff_input)
        tariff_input.set_value(12.50).run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Exceptions in Usage Patterns on tariff change: {at.exception}")
        self.assertAlmostEqual(at.session_state.get("tariff_rate", 0.0), 12.50, places=2)


if __name__ == "__main__":
    unittest.main()
