#!/bin/bash

TERMINAL_WINDOW_ID=$(xdotool getwindowfocus)

for file in ./dataset/car_cut/*; do
    echo "File: $file"
    echo "Click 'f' to move file to front, 'b' to back, 's' to side, 'q' to quit"


    feh -g 800x600 -x "$file" &

    FEH_PID=$!

    sleep 0.5

    xdotool windowfocus $TERMINAL_WINDOW_ID

    read -n 1 key

    kill $FEH_PID

    case $key in
        f)
            mv "$file" /datasets_labeled/front/images/
            ;;
        b)
            mv "$file" /datasets_labeled/back/images/
            ;;
        t)
            rm "$file"
            ;;
        s)
            mv "$file" /datasets_labeled/side/images/
            ;;
        q)
            exit 0
            ;;
        *)
            echo "Unknown input: $key"
            ;;
    esac
done
