@ECHO OFF

pushd %~dp0

REM Sphinx 文档的命令文件

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
    echo.
    echo.未找到 'sphinx-build' 命令。请确保已安装 Sphinx，
    echo.然后将 SPHINXBUILD 环境变量设置为指向
    echo.'sphinx-build' 可执行文件的完整路径。或者你
    echo.可以将 Sphinx 目录添加到 PATH。
    echo.
    echo.如果你没有安装 Sphinx，请从以下网址获取
    echo.http://sphinx-doc.org/
    exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
