#!/bin/bash
echo Script name: $0
echo $# arguments 
if [ $# -ne 2 ]; 
    then echo "illegal number of parameters"
	echo "Usage : make_video.sh path_with_% output_file.mp4"
	echo "	Example : make_video.sh img/%04d.png output_file.mp4"
	echo "	This means the images will have pattern img/0000.png, img/0001.png, ..."
	exit
fi
ffmpeg -f image2 -r 10 -i $1 -vcodec mpeg4 -y $2
