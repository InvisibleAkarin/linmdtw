if [[ "$DISTRIB" == "conda" ]]; then

  # 要运行 conda python，请使用以下参数
  #  DISTRUB="conda"
  #  TRAVIS_OS_NAME=["linux", "osx"]
  #  PYTHON_VERSION=["3.4", "3.5", "3.6", "3.7"]

  # 停用 travis 提供的虚拟环境，并改用基于 conda 的环境
  deactivate

  # 使用 miniconda 安装程序更快地下载/安装 conda 本身
  pushd .
  cd
  mkdir -p download
  cd download
  echo "缓存于 $HOME/download :"
  ls -l
 
  if [ "$TRAVIS_OS_NAME" = linux ]; then
      sudo apt-get update
      MINICONDAVERSION="Linux"
  else
      MINICONDAVERSION="MacOSX"
  fi

  echo "设置 miniconda 版本 $MINICONDAVERSION"

  echo "获取 python 3 的 miniconda"
  if [[ ! -f miniconda.sh ]]; then
      wget https://repo.continuum.io/miniconda/Miniconda3-latest-$MINICONDAVERSION-x86_64.sh -O miniconda.sh 
  fi

  MINICONDA_PATH=$HOME/miniconda
  chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
  export PATH=$MINICONDA_PATH/bin:$PATH

  cd $TRAVIS_BUILD_DIR

  # 使用提供的版本配置 conda 环境并将其放入路径中
  conda create -n testenv --yes python=$PYTHON_VERSION numpy matplotlib numba scipy pip cython pytest pytest-cov

  source activate testenv

  python --version
  python -c "import numpy; print('numpy %s' % numpy.__version__)"
  python -c "import scipy; print('scipy %s' % scipy.__version__)"
  python setup.py install



else
  pip3 install numpy matplotlib numba scipy pip cython pytest pytest-cov
  pip3 install .
  pip3 install pytest-cov
fi