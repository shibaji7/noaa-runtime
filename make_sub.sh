cp -r ~/dst_predictor/algorithms/LR/model/* benchmark/algorithms/LR/
cp -r ~/dst_predictor/algorithms/LSTM/model/* benchmark/algorithms/LSTM/
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
rm -rf submission/*
sudo make pack-benchmark
#sudo make test-submission
