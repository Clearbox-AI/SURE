rm -rf ./dist
rm -rf ./build

pip install -r requirements.txt
python3 setup.py build_ext 
python3 setup.py bdist_wheel
pip install --force-reinstall dist/*.whl

