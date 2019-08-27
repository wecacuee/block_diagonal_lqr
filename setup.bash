D=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
# System dependencies
if [ ! -f $D/.installed-apt-get-requirements.flag ]; then
    sudo apt-get update && \
        sudo apt-get install -y $(cat apt-get-requirements.txt)
    touch $D/.installed-apt-get-requirements.flag
fi
# Python dependencies
if [ ! -f $D/block_diagonal_lqr.egg-info/PKG-INFO ]; then
    pip install -e $D
    touch $D/block_diagonal_lqr.egg-info/PKG-INFO
fi
[ -f envrc ] && source envrc
