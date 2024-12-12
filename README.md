# LAIA-SQL

## Environment Setup

input the following code in terminal:

```
conda create -n laia python==3.10
conda activate laia
pip install -r requirements.txt
```


## Model Download

Step by step, input the following content in terminal:

```
huggingface-cli login
```

```
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
```

```
huggingface-cli download dunzhang/stella_en_400M_v5 --local-dir stella_en_400M_v5
```

## Preprocess Test Data

step 1: put all the test data (test.json, test_databases folder) into the folder: data/test.

step 2: run the code in terminal:

```
sh run/run_preprocess.sh
```

## Code Generation

step 1: run the code in terminal:

```
sh run/run_main.sh
```

step 2: check the result in the folder: result
