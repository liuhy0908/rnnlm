#########################################################################
# File Name: run.sh
# Author:GuMJ
# mail: gmj_java@163.com
# Created Time:二  3/ 1 15:44:33 2016
TASKNAME=test_lm
LOGFILE=${TASKNAME}.log
ERRFILE=${TASKNAME}.err
BINNAME=../src/language_model.py
LINKNAME=${TASKNAME}.link
ln -s $BINNAME $LINKNAME
THEANO_FLAGS=device=gpu0,floatX=float32 nohup python $LINKNAME 1>>$LOGFILE 2>$ERRFILE & 
