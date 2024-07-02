# Sphinx 文档的最小 Makefile
#

# 你可以从命令行设置这些变量，也可以
# 从环境中设置前两个变量。
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# 把它放在最前面，这样没有参数的 "make" 就像 "make help" 一样。
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# 通用目标：使用新的 "make 模式" 选项将所有未知目标路由到 Sphinx。$(O) 是 $(SPHINXOPTS) 的简写。
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
