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


# Clone Rollout methods
echo -e "\e[1;32m[-] Cloning Attention Rollout method\e[0m"
git clone https://github.com/jacobgil/vit-explain vit_explain
cd vit_explain
git -c advice.detachedHead=false checkout 15a81d355a5aa6128ea4e71bbd56c28888d0f33b
cd ..
echo ""


# Clone TAM method
echo -e "\e[1;32m[-] Cloning Transition Attention Maps method\e[0m"
git clone https://github.com/XianrenYty/Transition_Attention_Maps
cd Transition_Attention_Maps
git -c advice.detachedHead=false checkout 8329327179f6dcf76df29e03ae519cd594595678
cd ..
echo ""

# Clone BT method
echo -e "\e[1;32m[-] Cloning method from 'Beyond Intuition: Rethinking Token Attributions inside Transformers'\e[0m"
git clone https://github.com/jiaminchen-1031/transformerinterp
cd transformerinterp
git -c advice.detachedHead=false checkout f710ac8cac1f84e8e53c2f7fb3ec4782de26dfce
cd ..
echo ""


# Clone ViT-CX
echo -e "\e[1;32m[-] Cloning ViT-CX method\e[0m"
git clone https://github.com/vaynexie/CausalX-ViT ViTCX
cd ViTCX
git -c advice.detachedHead=false checkout a120d99346d5d01788c221513879d1a6a5c70a39
cd ..
echo ""

echo -e "\e[1;32m[-] Done"

