#!/bin/bash
echo Script name: $0
echo $# arguments 
if [ $# -ne 2 ]; 
    then echo "illegal number of parameters"
	echo "Usage : make_video.sh path_with_% output_file.mp4"
	exit
fi
ffmpeg -f image2 -r 10 -i $1 -vcodec mpeg4 -y $2
