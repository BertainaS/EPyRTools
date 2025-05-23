"""
Isotopes GUI Package

This package provides a GUI for viewing isotope data.
"""

# Import specific, useful components from the isotopes_gui module
# The '.' means "from the current package"
from .isotope_gui import run_gui as isotopes
from .eprload import *
from .constants import *
from .plot import *
from .sub.baseline import *
from .sub.utils import BrukerListFiles
from .Trash.fair4 import *

# You can add package-level initialization code here if needed in the future.
print(f"Package 'your_project_folder' initialized.") # Optional: confirmation message