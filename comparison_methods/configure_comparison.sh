#!/bin/bash

# Get script path and go there (in case script is lauched from another dir)
SWD="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SWD

echo -e "    \e[32m ______________________________________\e[0m"
echo -e "    \e[32m|                                      |\e[0m"
echo -e "    \e[32m| Configuration script for comparisons |\e[0m"
echo -e "    \e[32m|______________________________________|\e[0m"
echo ""
echo ""
echo -e "\e[1;32mDownloading comparisons methods\e[0m"
echo ""

# For each method, the commit used in experiments is specified to avoid problem 
# in case of futur modifications of a repository

# Clone RISE
echo -e "\e[1;32m[-] Cloning RISE\e[0m"
git clone https://github.com/eclique/RISE RISE 
cd RISE
git -c advice.detachedHead=false checkout d91ea006d4bb9b7990347fe97086bdc0f5c1fe10
cd ..
echo ""


# Clone Chefer methods
echo -e "\e[1;32m[-] Cloning Chefer's methods\e[0m"
git clone https://github.com/hila-chefer/Transformer-Explainability chefer1
cd chefer1
git -c advice.detachedHead=false checkout 72ba6816b90c6c595eac6cb2be09e6cfe2218a7b
cd ..
echo ""

echo -e "\e[1;32m[-] Done"

