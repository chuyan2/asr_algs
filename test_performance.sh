./run_test.sh $1 &
pid=$(pgrep -f test.py)
python test_performance.py $pid &
