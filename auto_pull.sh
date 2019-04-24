#!/bin/sh
cd "${0%/*}"

git fetch
LOCAL=$(git rev-parse HEAD)
UPSTREAM=$(git rev-parse @{u})
if [ $LOCAL = $UPSTREAM ]; then
	echo same
	echo $LOCAL
	echo $UPSTREAM
else
	echo changed
	echo $LOCAL
	echo $UPSTREAM
	git pull
	cd ./python/
	python setup.py vuild_ext --inplace
fi

