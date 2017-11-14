LOG_FILE=$1
DIR=$(dirname $LOG_FILE)
TEMP_DIR=$DIR/temp_logs
mkdir $TEMP_DIR
/opt/flownet2/tools/extra/parse_log.py $LOG_FILE $TEMP_DIR
#python3 plot_logs.py $TEMP_DIR/log.txt.train $TEMP_DIR/log.txt.test
