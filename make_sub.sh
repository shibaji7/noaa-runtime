cp -r /home/data/dst_pred_data/algorithms benchmark/
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
rm -rf submission/*
sudo make pack-benchmark
sudo make test-submission
