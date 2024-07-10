# Remove existing build and distribution directories
rm -rf ./dist
rm -rf ./build

# Install required packages
pip install -r requirements.txt

# Build the package
python3 setup.py build_ext 
python3 setup.py bdist_wheel

# Install the built package
pip install --force-reinstall dist/*.whl
