#!/bin/bash

if [ ! -d ./san-manager ]; then
	echo "[*] Clone gensim"
	git clone git@github.com:piskvorky/gensim.git lib/gensim
fi
