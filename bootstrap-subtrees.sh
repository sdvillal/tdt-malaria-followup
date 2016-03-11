#!/bin/bash

git init
git add .gitignore
git commit -m 'generic ignores'

git add bootstrap-subtrees.sh
git commit -m 'script to bootstrap original submission subtrees'

subtreemerge(){
  # https://www.kernel.org/pub/software/scm/git/docs/howto/using-merge-subtree.html
  local name=${1}
  local url=${2}
  git remote add -f ${name} ${url}
  git merge -s ours --no-commit ${name}/master
  git read-tree --prefix=submissions/${name} -u ${name}/master
  git commit -m "subtree merged in ${name}"
  echo "Update subtree with: git pull -s subtree $name master"
}

subtreemerge "sereina" "git@github.com:sriniker/TDT-tutorial-2014.git"
subtreemerge "santi" "git@github.com:sdvillal/ccl-malaria.git"

git remote add origin git@github.com:sdvillal/tdt-malaria-followup.git
git push -u origin master