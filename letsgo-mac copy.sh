#!/usr/bin/env sh

echo I assume you have Python and Jupyter installed and accessible in this environment

CURRENT_HOME="$( cd "$(dirname "$0")" ; pwd -P )"
export PYSPARK=$voila/CURRENT_HOME/MDA_thibaut_1.pynb

echo Ready to start
echo $PYSPARK

$PYSPARK
