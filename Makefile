
.PHONY: style test docs

check_dirs := tfod examples tests

# run checks on all files and potentially modifies some of them

style:
	black --preview $(check_dirs)
	isort $(check_dirs)
	flake8
	pre-commit run --all-files

# run tests for the library

test:
	python -m unittest

# run tests for the docs

docs:
	make -C docs clean M=$(shell pwd)
	make -C docs html M=$(shell pwd)

# release

.PHONY: initgit
initgit:
	git init
	git add .
	git commit -m "initial"
	# git remote rm origin
	git remote add origin "git@github.com:yuetan1988/tf-outlier.git"
	git push -u origin master
