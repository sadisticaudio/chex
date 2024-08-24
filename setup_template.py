from setuptools import setup, Extension
import os

pyVer = "{PYVERSION}"
pyName = "{PYNAME}"

# Get the directory containing the setup.py file
current_dir = os.path.abspath(os.path.dirname(__file__))
current_library_dir = os.path.join(current_dir, 'lib')
current_src_dir = os.path.join(current_dir, 'src')
cuda_home = '/usr/local/cuda'
einops_home = '/media/frye/sda5/CODE/einops-cpp'

os.environ['EINOPS_TORCH_BACKEND'] = "True"

# Define the include directories, library directories, and libraries
include_dirs = [current_src_dir, '/media/frye/sda5/anaconda3/envs/pytorch_env/lib/python3.10/site-packages/torch/include/torch/csrc/api/include', '/media/frye/sda5/anaconda3/envs/pytorch_env/lib/python3.10/site-packages/torch/include', '/media/frye/sda5/boost_1_85_0/include', cuda_home + '/include', einops_home + '/include']
library_dirs = [current_library_dir, '/media/frye/sda5/boost_1_85_0/stage/lib']
libraries = ['Chex' + pyName, 'boost_python' + pyName, 'boost_system']  # Add 'boost_python' if needed

# Define the extension module
extension_mod = Extension(
    name='chex',
    sources=[os.path.join(current_src_dir, 'chex.cpp')],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=['-std=c++2b', '-fPIC', '-fno-omit-frame-pointer', '-pthread', '-Wno-deprecated-declarations'],  # Specify any extra compilation flags if needed
    extra_link_args=['-Wl,-rpath,' + '/media/frye/sda5/boost_1_85_0/stage/lib'],  # Set RPATH to Boost libraries
)

# Setup
setup(
    name='chex',
    version='0.0.1',
    ext_modules=[extension_mod],
    exclude_package_data={'': ['junk']},
)

# from setuptools import setup, find_packages, Extension

# setup(
#     name='example-boost-python',
#     version='0.0.1',
#     description='Example for Boost.Python',
#     author='Benjamin Kelley',
#     author_email='your-name@example.com',
#     url='https://example.com/your-repo',
#     install_requires=[],
#     tests_require=[],
#     package_dir={'': 'src'},
#     packages=find_packages('src'),
#     include_package_data=True,
#     test_suite='tests',
#     entry_points='',
#     ext_modules=[
#         Extension(
#             name='chex',
#             sources=['src/chello/chello.cpp'],
#             include_dirs=['/usr/local/include/boost'],
#             library_dirs=['/usr/lib', '/usr/local/lib'],
#             libraries=['boost_python3'],
#             extra_compile_args=['-std=c++11', '-Wall'],
#             extra_link_args=[],
#         )
#     ],
# )