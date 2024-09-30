# Runtime Integration of Machine Learning and Simulation for Business Processes (RIMS) - Recreated Results

This repository contains the recreated results for the paper _Runtime Integration of Machine Learning and Simulation for Business Processes_. We have successfully replicated all results for the RIMS and RIMS+ models. However, there were issues when reproducing some results for the other models, which we believe are due to incomplete or faulty datasets provided to us.

### Known Issues:

We encountered errors while running the LSTM and DSIM models on certain datasets:

- **BPI_Challenge_2012_W_Two_TS**:
  - Error running LSTM
  - Error running DSIM
- **PurchasingExample**:
  - Error running LSTM
  - Error running DSIM
- **Productions**:
  - Error running LSTM
  - Error running DSIM

We think this is due to the data rather then out code, since it works for everything else.

### Repository Structure:

The code is divided into two sections:

1. **RIMS and RIMS+**
2. **DDPS, DSIM, and LSTM**

### Before running

Notice that due to github size limit, we had to remove the Logs folder from the OTHER directory. it can be found here:

https://drive.google.com/drive/folders/1gmO8ULxtBxqShXnBeEUhBLOy97KYlVI2

Just download and copy and paste the Log folder into OTHER/

### Running the Models:

To run the **RIMS and RIMS+** models:

```bash
cd RIMS_RIMS_PLUS
python run.py
```

To run the **DDPS, DSIM, and LSTM** models:

```bash
cd OTHER
python run.py
```
