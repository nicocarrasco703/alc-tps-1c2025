#!/usr/bin/env bash

entorno=envtp1

deactivate

if [ -d "$entorno" ]; then
  rm -rf $entorno
  jupyter kernelspec uninstall $entorno
fi
