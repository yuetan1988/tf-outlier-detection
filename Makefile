
.PHONY: initgit
initgit:
	git init
	git add .
	git commit -m "initial"
	# git remote rm origin
	git remote add origin "git@github.com:yuetan1988/tf-outlier.git"
	git push -u origin master
