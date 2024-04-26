#!/bin/bash

DATASET_PATH="./dataset/car_cut"
FRONT_PATH="./datasets_labeled/front"
BACK_PATH="./datasets_labeled/back"
SIDE_PATH="./datasets_labeled/side"

TERMINAL_WINDOW_ID=$(xdotool getwindowfocus)

a=0

for file in "$DATASET_PATH"/*; do
    echo "File: $file"
    echo "Click 'f' to move file to front, 'b' to back, 's' to side, 'q' to quit"


    feh --auto-zoom -x -g +0+0 "$file" &

    FEH_PID=$!

    sleep 0.5

    xdotool windowfocus $TERMINAL_WINDOW_ID

    read -n 1 key

    kill $FEH_PID

    case $key in
        f)
            mv "$file" "$FRONT_PATH"/
            ;;
        b)
            mv "$file" "$BACK_PATH"/
            ;;
        t)
            rm "$file"
            ;;
        s)
            mv "$file" "$SIDE_PATH"/
            ;;
        q)
            exit 0
            ;;
        *)
            echo "Unknown input: $key"
            ;;
    esac
    ((a++))
    echo $a
done
