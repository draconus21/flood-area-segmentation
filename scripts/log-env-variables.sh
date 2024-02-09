#! /bin/bash

normal=$(tput sgr0)
red=$(tput setaf 1)
green=$(tput setaf 2)
yellow=$(tput setaf 3)
cyan=$(tput setaf 6)

function green() {
  echo -e ${green}$1${normal}
}

function red() {
  echo -e ${red}$1${normal}
}

function yellow() {
  echo -e ${yellow}$1${normal}
}

function cyan() {
  echo -e ${cyan}$1${normal}
}

DEFAULT_LD_LIBRARY_PATH=""
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-$DEFAULT_LD_LIBRARY_PATH}

numbersOfEnvVarsFound=$(env | grep -c '^FSEG_')
yellow "[FSEG env vars] ${cyan}# = ${normal}$numbersOfEnvVarsFound"
env | grep '^FSEG_' | GREP_COLOR='1;34' grep --color=always -P '^[^=]+' | sort

numbersOfPathsFound=$(sed 's/:/\n/g' <<<"$PATH" | grep -c -E "$OpenCV_DIR")
yellow "[PATH] ${cyan}# = ${normal}$numbersOfPathsFound"
sed 's/:/\n/g' <<<$PATH | grep -E "$OpenCV_DIR"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  if [[ -n $LD_LIBRARY_PATH ]]; then
    numbersOfLDLibraryPathsFound=$(sed 's/:/\n/g' <<<"$LD_LIBRARY_PATH" | grep -c -E "$OpenCV_DIR")
    yellow "[LD_LIBRARY_PATH] ${cyan}# = ${normal}$numbersOfLDLibraryPathsFound"
    sed 's/:/\n/g' <<<$LD_LIBRARY_PATH | grep -E "$OpenCV_DIR"
  fi
fi
