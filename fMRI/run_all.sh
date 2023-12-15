#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <class1> <class2> <num_threads>"
  exit 1
fi

class1="$1"
class2="$2"
num_threads="$3"


python global_script.py lasso "$class1" "$class2" "$num_threads" &
python global_script.py enet "$class1" "$class2" "$num_threads" &
python global_script.py omp "$class1" "$class2" "$num_threads" &
python global_script.py iht "$class1" "$class2" "$num_threads" &
python global_script.py ksn "$class1" "$class2" "$num_threads" &
python global_script.py irksn "$class1" "$class2" "$num_threads" &
python global_script.py ircr "$class1" "$class2" "$num_threads" &
python global_script.py irosr "$class1" "$class2" "$num_threads" &
python global_script.py srdi "$class1" "$class2" "$num_threads" &
python encludl.py "$class1" "$class2" "$num_threads" &
wait

