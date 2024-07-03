import sys
import os
import platform

from setuptools import setup
from setuptools.extension import Extension

# 确保在尝试安装 linmdtw 之前已安装 Cython
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    print("你似乎没有安装 Cython。请从 www.cython.org ")
    print("获取副本或使用 `pip install Cython` 安装它")
    sys.exit(1)

## 从 _version.py 获取版本信息
import re
VERSIONFILE="linmdtw/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("无法在 %s 中找到版本字符串。" % (VERSIONFILE,))

# 使用 README.md 作为包的长描述
with open('README.md') as f:
    long_description = f.read()

class CustomBuildExtCommand(build_ext):
    """ 这个扩展命令让我们在运行 pip 安装之前不需要安装 numpy
        build_ext 命令用于需要 numpy 头文件时。
    """

    def run(self):
        # 仅在需要头文件时在此处导入 numpy
        import numpy
        # 将 numpy 头文件添加到 include_dirs
        self.include_dirs.append(numpy.get_include())
        # 调用原始的 build_ext 命令
        build_ext.run(self)

extra_compile_args = []
extra_link_args = []

if platform.system() == "Darwin":
    extra_compile_args.extend([
        "-mmacosx-version-min=10.9"
    ])
    extra_link_args.extend([
        "-mmacosx-version-min=10.9"
    ])

# ext_modules = Extension(
#     "dynseqalign",
#     sources=["linmdtw/dynseqalign.pyx"],
#     define_macros=[
#     ],
#     extra_compile_args=extra_compile_args,
#     extra_link_args=extra_link_args,
#     language="c++"
# )
ext_modules = Extension(
    "linmdtwPy2Cpp",
    sources=["linmdtw/linmdtwPy2Cpp.pyx"],
    define_macros=[
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++"
)


setup(
    name="linmdtw",
    version=verstr,
    description="A linear memory, exact, parallelizable algorithm for dynamic time warping in arbitrary Euclidean spaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chris Tralie",
    author_email="ctralie@alumni.princeton.edu",
    license='Apache2',
    packages=['linmdtw'],
    ext_modules=cythonize(ext_modules, include_path=['linmdtw']),
    install_requires=[
        'Cython',
        'numpy',
        'matplotlib',
        'scipy',
        'numba'
    ],
    extras_require={
        'testing': [ # `pip install -e ".[testing]"``
            'pytest'  
        ],
        'docs': [ # `pip install -e ".[docs]"`
            'linmdtw_docs_config'
        ],
        'examples': []
    },
    cmdclass={'build_ext': CustomBuildExtCommand},
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    keywords='dynamic time warping, alignment, fast dtw, synchronization, time series, music information retrieval, audio analysis'
)
