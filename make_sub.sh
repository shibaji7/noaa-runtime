find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
rm -rf submission/*
sudo make pack-benchmark
sudo make test-submission
