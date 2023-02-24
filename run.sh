#!/bin/bash

command=$1

if [ "$command" = "help" ]; then
    echo "./run.sh [commands]" 
    echo "./run.sh run : \"Run flask project\"" 
elif [ "$command" = "run" ]; then
    flask run
fi

