#! /bin/bash

ABOUT_SCRIPT="
Flood Area Segmentation: Expriments with Kaggle Flood Area Segmentation dataset (https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation).

Setup dev environment for Flood Area Segmentation.
Make sure to run this script in each terminal you want to develop with Flood Area Segmentation."

# ------------------------------------------------------------------- #
#                          TERMINAL COLORS                            #
# ------------------------------------------------------------------- #

normal=$(tput sgr0)
bg_normal=$(tput setab sgr0)
bg_black=$(tput setab 0)
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

# ------------------------------------------------------------------- #
#                            HELPER FUNCTIONS                         #
# ------------------------------------------------------------------- #

# Adds visual padding for visibility
# usage - output_spacing "command" "info"
tmp_padding="                                    " # expand as necessary...
function output_spacing() {
  tmp_stringToPad=$1
  printf "%s%s %s %s\n" "${yellow}$1" "${tmp_padding:${#tmp_stringToPad}}" ":" "${normal}$2"
}

# outputs received script arguments
# SYNTAX - output_received_args "$@"
function output_received_args () {
  echo -n "$bg_black""$yellow"  # set bg/text colors
  args=("$@")           # store arguments in a special array
  ELEMENTS=${#args[@]}  # get number of elements

  printf "[cmd] : ${args}"

  echo -n "$bg_normal""$normal" # reset bg/text colors
  printf "\n\n"
}

# exports an array of env. vars
# usage - export_env_var_arrays "array"
function export_env_var_arrays() {
	arr=("$@")
	for i in "${arr[@]}"; do
		export "${i?}"
	done
}

# displays an array of env. vars
# usage - display_env_var_arrays "array"
function display_env_var_arrays() {
	arr=("$@")

	for i in "${arr[@]}"; do
		case $SHELL in
      */zsh) # shell-check doesn't support zsh and will mark as error
      output_spacing "${i}" "${(P)i}"
      ;;
      */bash) # ${!i} is incompatible on zsh (indirect expansion)
      output_spacing "${i}" "${!i}"
      ;;
      *)
      echo "no compatible shells"
		esac

	done
}

make_dir() {
  if [[ ! -d $1 ]]; then
    echo "creating dir: $1"
    mkdir -p $1
  fi
}

function setup_env() {
  green "═══ floodsegment env.sh ═══"

  USER_VENV=${1:-""}
  DEFAULT_VENV=".env"



  # ------------------------------------------------------------------- #
  #                                  ARGS                               #
  # ------------------------------------------------------------------- #
  FSEG_DIR="$(pwd)"

  # ------------------------------------------------------------------- #
  #                             BASE Env.Vars                           #
  # ------------------------------------------------------------------- #

  if [[ ! -d "$FSEG_DIR" ]]; then
    red "Error: FSEG_DIR:$FSEG_DIR doesn't point to a valid directory"
    return 1
  fi

  arrayEnvVarsToExport=(FSEG_DIR)

  export_env_var_arrays "${arrayEnvVarsToExport[@]}"
  display_env_var_arrays "${arrayEnvVarsToExport[@]}"
  chmod u+x -R "${FSEG_DIR}/scripts"

  # ------------------------------------------------------------------- #
  #                                 PYTHON                              #
  # ------------------------------------------------------------------- #
  green "\n--- Python Env.Vars ---"

  if [[ "$OSTYPE" == "msys" ]]; then
    FSEG_PYTHON_EXECUTABLE=$(which python)
  else
    FSEG_PYTHON_EXECUTABLE=$(which python3)
  fi

  FSEG_PYTHON_VENV="${USER_VENV:=${FSEG_DIR}/${DEFAULT_VENV}}"

  if [[ "$OSTYPE" == "msys" ]]; then
    FSEG_PYTHON_VENV_PATH="${FSEG_PYTHON_VENV}"/Scripts/activate
  else
    FSEG_PYTHON_VENV_PATH="${FSEG_PYTHON_VENV}"/bin/activate
  fi

  # check if venv dir exists, if not create one after confirming with user
  if [[ ! -d ${FSEG_PYTHON_VENV} ]]; then
    red "virtual env does not exist at ${FSEG_PYTHON_VENV}"
    case $SHELL in
    */zsh)
      vared -p "Would you like me to create one? [y/n]: " -c confirm
      ;;
    */bash) # vared incompatible on bash
      echo "Would you like me to create one? [y/n]: "
      read confirm
      ;;
    *)
      echo "no compatible shells"
      ;;
    esac
    if [[ "$confirm" == "y" ]]; then
      yellow "creating venv ${FSEG_PYTHON_VENV}"
      "${FSEG_PYTHON_EXECUTABLE}" -m venv "${FSEG_PYTHON_VENV}"
    fi
  fi

  source ${FSEG_PYTHON_VENV_PATH}

  # get python executable from venv
  FSEG_PYTHON_EXECUTABLE=$(which python)
  FSEG_PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

  arrayEnvVarsToExport=(FSEG_PYTHON_VENV
    FSEG_PYTHON_VENV_PATH
    FSEG_PYTHON_EXECUTABLE
    FSEG_PYTHON_VERSION)

  export_env_var_arrays "${arrayEnvVarsToExport[@]}"
  display_env_var_arrays "${arrayEnvVarsToExport[@]}"

  green "\n--- Final Env.Vars ---"
  cyan "(All env. vars. generated from this script related to FSEG)"
  $FSEG_DIR/scripts/log-env-variables.sh

  green "═══ floodsegment env.sh ═══"
}

function FNC_help() {
  # usage - help_outputs "command" "info"
  tmp_padding="          "
  function help_outputs() {
    tmp_stringToPad=$1
    printf "%s%s %s %s\n" "${yellow}$1" "${tmp_padding:${#tmp_stringToPad}}" ":" "${normal}$2"
  }

  green "\n________________________ PURPOSE ________________________"
  echo "$ABOUT_SCRIPT"

  green "\n________________________ USAGE INFO ________________________"

  instruction=("SYNTAX" "${normal}source scripts/env.sh ${yellow}[venv-name]")
  printf "%s%s %s\n" "${green}${instruction[0]}" "${tmp_padding:${#instruction[0]}}" "${instruction[1]}"

  help_outputs "-h" "Print this message"
  help_outputs "venv-name" "Activates the python virtual environment ${cyan}venv-env${normal}"
  help_outputs "" "Additionally, if ${cyan}venv-env${normal} does not exist, it will prompt the user and create one."
  help_outputs "" "This is an OPTIONAL argument. If no ${cyan}venv-env${normal} is provided,"
  help_outputs "" "it will create and/or activate the default venv (.env)"
  printf "\n"
}

output_received_args "$0" "$@"

# shifting over all --${OPTARG}s
# shift $((OPTIND - 1))
DEFAULT_SCRIPT_ARGUMENT=""
SCRIPT_ARGUMENT=${1:-${DEFAULT_SCRIPT_ARGUMENT}}

case "${SCRIPT_ARGUMENT}" in
-h | -help | --help | --h)
  FNC_help
  ;;
*)
  setup_env $SCRIPT_ARGUMENT
  ;;
esac

