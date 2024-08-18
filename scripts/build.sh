#! /bin/bash

# ------------------------------------------------------------------- #
#                              ABOUT                                  #
# ------------------------------------------------------------------- #
ABOUT_SCRIPT="
Scripts to setup and clean floodsegment."

# ------------------------------------------------------------------- #
#                          TERMINAL COLORS                            #
# ------------------------------------------------------------------- #

normal=$(tput sgr0)
red=$(tput setaf 1)
yellow=$(tput setaf 3)
green=$(tput setaf 2)
cyan=$(tput setaf 6)

function green() {
    echo ${green}$1${normal}
}

function yellow() {
    echo ${yellow}$1${normal}
}

function red() {
    echo ${red}$1${normal}
}
function cyan() {
    echo -e ${cyan}$1${normal}
}

# ------------------------------------------------------------------- #
#                          HELPER FUNCTIONS                           #
# ------------------------------------------------------------------- #

function error() {
    red "$@"
    exit -1
}

function rmdir() {
    if [[ -z "$1" ]]; then
        red "var ${1} not set"
        return -1 # fail
    elif [[ ! -d "$1" ]]; then
        red "directory not found: ${1}"
        return 0 # no fail; continue w/o deleting
    else
        yellow "removing $1"
        rm -rf "$1"
    fi
}

if [[ -z "${FSEG_DIR}" ]]; then
    FSEG_DIR=$(pwd)
    error "FSEG_DIR is empty. Please run ${yellow}source ./scripts/env.sh${red} from the repo's root directory."
fi

# -o prevent errors from being masked
# -u require vars to be declared before referencing them
set -uo pipefail

function clean_whl() {
    cyan "[clean-whl] start"
    rmdir "${FSEG_DIR}/dist"
    rmdir "${FSEG_DIR}/build"
    cyan "[clean-whl] done"
}

function clean_env() {
    cyan "[clean-env] start"
    if [[ -n "${FSEG_PYTHON_VENV}" ]]; then
        if [[ -d "${FSEG_PYTHON_VENV}" ]]; then
            cyan "removing your virtual env folder: ${FSEG_PYTHON_VENV}"
            rmdir "${FSEG_PYTHON_VENV}"/
        fi
    fi
    cyan "[clean-env] done"
    yellow "Don't forget to run deactivate from your terminal."
}

function build_whl() {
    cyan "[build-whl] start"
    python -m build ${FSEG_DIR}
    cyan "[build-whl] done"
}

function install_dev() {
    cyan "[install-dev] start"

    # install package locally
    pip install --upgrade pip || red "caught pip upgrade error"
    curdir="$(pwd)"
    cd $FSEG_DIR
    pip install -e .[dev]
    cd $curdir
    cyan "[install-dev] end"
}

function FNC_help() {
    # usage - help_outputs "command" "info"
    tmp_padding="                                "
    function help_outputs() {
        tmp_stringToPad=$1
        printf "%s%s %s %s\n" "${yellow}$1" "${tmp_padding:${#tmp_stringToPad}}" ":" "${normal}$2"
    }

    green "\n________________________ PURPOSE ________________________"
    echo "$ABOUT_SCRIPT"

    green "\n________________________USAGE INFO________________________"

    instruction=("SYNTAX" "${normal}build.sh ${cyan}<options set A>")
    printf "%s%s %s\n" "${green}${instruction[0]}" "${tmp_padding:${#instruction[0]}}" "${instruction[1]}"

    cyan "\n[options set A]"
    help_outputs "-h" "Print this message"
    help_outputs "dev" "pip install -e .[dev]"
    help_outputs "" "install -e allows for live editing of code, without a pip re-install"
    help_outputs "wheel" "builds floodsegment whl"
    help_outputs "" "generates dist/floodsegment-....whl (build distribution) and dist/floodsegment....tar.gz (source distribution)"
    help_outputs "clean-env" "cleans up the virtual Python environment"
    help_outputs "" "[${FSEG_PYTHON_VENV}]"
    help_outputs "clean-whl" "cleans up folders created during ./scripts/build.sh wheel"
    help_outputs "clean" "runs clean-whl and clean-env"

    cyan "\n[What is the diff between source/build distributions?]"
    help_outputs "Source Distributions" "Packages in .tar.gz that is platform AGNOSTIC that allows the user to build from source"
    help_outputs "Build Distributions" "Packages in .whl CAN BE platform SPECIFIC"
    help_outputs "More info" "https://sinclert.github.io/packaging/"

    printf "\n"
}

SCRIPT_ARG=${1:-""}

case "${SCRIPT_ARG}" in
clean-env)
    clean_env
    ;;
clean-whl)
    clean_whl
    ;;
clean)
    clean_whl
    clean_env
    ;;
dev)
    install_dev
    ;;
wheel)
    build_whl
    ;;
-h | --h | -help | --help | *)
    FNC_help
    exit 0
    ;;
esac
