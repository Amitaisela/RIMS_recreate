import subprocess

# run python main.py -t DDPS -l confidential_2000 -n 1
datasets = ["BPI_Challenge_2012_W_Two_TS","ConsultaDataMining201618","PurchasingExample","confidential_1000",
            "cvs_pharmacy","BPI_Challenge_2017_W_Two_TS", "Productions", "SynLoan", "confidential_2000"]

models = ["DDPS", "LSTM", "DSIM"]

for dataset in datasets:
    for model in models:
        print(f"Running {model} on {dataset}")
        result = subprocess.run(["python", "main.py", "-t", model, "-l", dataset, "-n", "1"])
        if result.returncode == 0:
            with open("log_success.txt", "a") as f:
                f.write(f"Ran {model} on {dataset}\n")
        else:
            with open("log_error.txt", "a") as f:
                f.write(f"Error running {model} on {dataset}\n")
        print(f"Finished {model} on {dataset}")

print("Finished all simulations")
