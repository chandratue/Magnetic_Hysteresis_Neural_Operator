## Codes for the paper: “Magnetic Hysteresis Modeling with Neural Operators”, IEEE Transactions on Magnetics

Repository Structure

The repository is organized into two main folders: src, and utils. Below is a detailed description of each folder and its contents.

src/

The src folder is divided into three subfolders: training, testing, and models.

    training/
        Contains the code for training models on different datasets. It is further divided into:
            FORC/: Scripts for training models on first-order reversal curves (FORCs).
                Example: RIFNO_train.py for training the RIFNO model on FORCs.
            minor_loops/: Scripts for training models on minor loops.
                Example: RIFNO_train.py for training the RIFNO model on minor loops.

    testing/
        Contains the code for testing models on different datasets. It is further divided into:
            FORC/: Scripts for testing models on first-order reversal curves (FORCs).
                Example: RIFNO_test.py for testing the RIFNO model on FORCs.
            minor_loops/: Scripts for testing models on minor loops.
                Example: RIFNO_test.py for testing the RIFNO model on minor loops.

    models/
        Contains the class definitions for the different models used (e.g., DeepONet, FNO, WNO, RIFNO, LSTM, EDLSTM, GRU, RNN).

utils/

The utils folder contains utility scripts:

Running the Code

To run the codes, two folders should be created by the downstream user.

data/

The data folder should contain two subfolders and a script:

    raw/
        Contains the raw dataset.

    processed/
        Contains the processed dataset, which is split into training and testing data.

    data_process.py
        This script processes the raw dataset and saves the split data in the processed/ folder.

results/

The results folder will store outputs related to model training and evaluation:

    logs/
        Contains log files generated during the training of models.

    preds/
        Stores the prediction outputs from the models.

    trained_models/
        Contains the saved models after training.

To run the training and testing scripts, one needs Python 3 and PyTorch installed. Follow the instructions below to run the training and testing scripts.

Example: Training RIFNO for FORC

Navigate to the training directory for FORC:

cd src/training/FORC/

Run the training script:

python3 RIFNO_train.py

Example: Testing RIFNO for FORC

Navigate to the testing directory for FORC:

cd src/testing/FORC/

Run the testing script:

python3 RIFNO_test.py
