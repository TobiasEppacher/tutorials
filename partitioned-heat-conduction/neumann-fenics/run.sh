#!/usr/bin/env bash
set -e -u

. ../../tools/log.sh
exec > >(tee --append "$LOGFILE") 2>&1

python3 ../solver-fenics/heatHigherOrder.py Neumann

close_log
