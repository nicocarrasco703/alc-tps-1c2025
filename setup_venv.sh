#!/usr/bin/env bash

set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

entorno=envtp1

if [ ! -d "$entorno" ]; then
  echo "Creando entorno: $entorno"
  python3 -m venv $entorno
  source $entorno/bin/activate
  pip install -r requirements.txt
  ipython kernel install --user --name=$entorno
fi

if [ -d "$entorno" ]; then
  echo "$entorno ya existe."
  source $entorno/bin/activate
  $entorno/bin/jupyter-lab
fi
