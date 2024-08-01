import shutil
import os
import numpy as np
from pathlib import Path
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# Read the content of the README.md for the long_description metadata
with open("README.md", "r") as readme:
    long_description = readme.read()

# Parse the requirements.txt file to get a list of dependencies
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# List of files to exclude from Cythonization
EXCLUDE_FILES = [   
    "sure/utility.py", 
    "sure/__init__.py",   
    "sure/privacy/__init__.py",
    "sure/privacy/privacy.py" 
    "sure/report_generator/report_generator.py",
    "sure/report_generator/report_app.py",
    "sure/report_generator/__init__.py",
]

def get_extensions_paths(root_dir, exclude_files):
    """
    Retrieve file paths for compilation.

    Parameters
    ----------
    root_dir : str
        Root directory to start searching for files.
    exclude_files : list of str
        List of file paths to exclude from the result.

    Returns
    -------
    list of str or Extension
        A list containing file paths and/or Extension objects.

    """
    paths = []

    # Walk the directory to find .py and .pyx files
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if (
                os.path.splitext(filename)[1] != ".py"
                and os.path.splitext(filename)[1] != ".pyx"
            ):
                continue

            file_path = os.path.join(root, filename)

            if file_path in exclude_files:
                continue

            if os.path.splitext(filename)[1] == ".pyx":
                file_path = Extension(
                    root.replace("/", ".").replace("\\", "."),
                    [file_path],
                    include_dirs=[np.get_include()],
                )

            paths.append(file_path)

    return paths

class CustomBuild(build_ext):
    """
    Custom build class that inherits from Cython's build_ext.

    This class is created to override the default build behavior.
    Specifically, it ensures certain non-Cython files are copied
    over to the build output directory after the Cythonization process.
    """

    def run(self):
        """Override the run method to copy specific files after build."""
        # Run the original run method
        build_ext.run(self)

        build_dir = Path(self.build_lib)
        root_dir = Path(__file__).parent
        target_dir = build_dir if not self.inplace else root_dir

        # List of files to copy after the build process
        files_to_copy = [
            "sure/distance_metrics/__init__.py",
            "sure/distance_metrics/gower_matrix_c.pyx",
            "sure/utility.py",    
            "sure/__init__.py",    
            "sure/privacy/__init__.py",
            "sure/privacy/privacy.py",   
            "sure/report_generator/report_generator.py",
            "sure/report_generator/report_app.py",
            "sure/report_generator/__init__.py"
        ]

        for file in files_to_copy:
            self.copy_file(Path(file), root_dir, target_dir)

    def copy_file(self, path, source_dir, destination_dir):
        """
        Utility method to copy files from source to destination.

        Parameters
        ----------
        path : Path
            Path of the file to be copied.
        source_dir : Path
            Directory where the source file resides.
        destination_dir : Path
            Directory where the file should be copied to.

        """
        src_file = source_dir / path
        dest_file = destination_dir / path
        dest_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        shutil.copyfile(str(src_file), str(dest_file))

# Main setup configuration
setup(
    # Metadata about the package
    name="clearbox-sure",
    version="0.1.3",
    author="Clearbox AI",
    author_email="info@clearbox.ai",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clearbox-AI/SURE",
    install_requires=requirements,
    python_requires=">=3.7.0",
    
    # Cython modules compilation
    ext_modules=cythonize(
        get_extensions_paths("sure", EXCLUDE_FILES),
        build_dir="build",
        compiler_directives=dict(language_level=3, always_allow_keywords=True),
    ),
    
    # Override the build command with our custom class
    cmdclass=dict(build_ext=CustomBuild),

    # List of packages included in the distribution
    packages=find_packages(),  # Include all packages in the distribution
    include_package_data=True,
)
