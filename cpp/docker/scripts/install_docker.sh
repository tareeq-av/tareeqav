#!/usr/bin/env bash

set -xe

function setup_gpg() {
    sudo apt-get update
    
    sudo apt-get -y install \
	 apt-transport-https \
	 ca-certificates \
	 curl \
	 gnupg-agent \
	 software-properties-common
    
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

}

function setup_docker_x86_repo() {
    sudo add-apt-repository \
	 "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
}

function setup_docker_arm_repo() {
    sudo bash -c 'echo "deb [arch=arm64] https://download.docker.com/linux/ubuntu xenial edge" > /etc/apt/sources.list.d/docker.list'
}

function install_docker() {
    sudo apt-get update
    sudo apt-get install -y docker-ce
    sudo usermod -aG docker $USER
}

function install() {
    # the machine type, currently support x86_64, aarch64
    setup_gpg
    MACHINE_ARCH=$(uname -m)
    if [ "$MACHINE_ARCH" == 'x86_64' ]; then
	setup_docker_x86_repo
    elif [ "$MACHINE_ARCH" == 'aarch64' ]; then
	setup_docker_arm_repo
    else
	echo "Unknown machine architecture $MACHINE_ARCH"
	exit 1
    fi
    install_docker
}

case $1 in
    　　install)
	install
	;;
    uninstall)
	sudo apt-get remove docker docker-engine docker.io
	sudo apt-get purge docker-ce
	;;
    *)
	install
	;;
esac
