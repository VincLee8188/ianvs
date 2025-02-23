# Tutorial for Evaluating an Algorithm with KubeEdge-Ianvs (LLM Simple QA)
This tutorial provides a comprehensive guide to evaluating algorithms using **KubeEdge-Ianvs**, an open-source benchmarking framework for edge AI scenarios. By following the steps outlined in this guide, you will learn how to set up an environment, prepare datasets, configure custom evaluation metrics, and run benchmarking jobs to assess algorithm performance in edge computing contexts.

A key highlight of this tutorial is the use of **FinReportQA**, a **brand-new dataset** that I have constructed specifically for financial question-answering tasks. This dataset is designed to provide diverse, context-rich financial data for more accurate and meaningful evaluations. Additionally, we leverage **GPT-4o** to implement a custom scoring metric, ensuring a rigorous and domain-specific evaluation process.

## Required Resources

Before using this example, ensure that you have a suitable device ready. One machine is sufficient—such as a laptop or virtual machine—and a cluster is not required.

- 2 CPUs or more
- 1 GPU with at least 6GB memory (depends on the tested model)  
- 4GB+ free memory (varies based on algorithm and simulation settings)  
- 10GB+ free disk space (depends on model size)  
- Internet connection (for GitHub, PyPI, HuggingFace, etc.)  
- Python 3.8+ environment

## Step 1: Environment Setup

```bash
# 1. Create a new conda environment with Python>=3.8
conda create -n ianvs38 python=3.8

# 2. Activate the environment
conda activate ianvs38

# 3. Clone the Ianvs repository
git clone https://github.com/kubeedge/ianvs.git
cd ianvs

# 4. Install Sedna
pip install examples/resources/third_party/sedna-0.6.0.1-py3-none-any.whl

# 5. Install example dependencies
pip install -r examples/cloud-edge-collaborative-inference-for-llm/requirements.txt

# 6. Install Ianvs core dependencies
pip install -r requirements.txt

# 7. Install Ianvs
python setup.py install

# 8. Install ONNX
pip install onnx
```

## Step 2: Dataset Preparation

The data of simple-qa example structure is:

```
.
├── test_data
│   └── data.jsonl
└── train_data
    └── data.jsonl
```

`train_data/data.jsonl` is empty, and the `test_data/data.jsonl` is prepared as follows:

1. Convert the FinReportQA Data

Run the `make_dataset.py` script to convert the **FinReportQA** dataset into a JSON Lines file (`data.jsonl`) suitable for question-answer evaluation.

```bash
python make_dataset.py
```

2. Move the Prepared Data

Copy the generated `data.jsonl` into the `test_data` folder of the LLM QA example:

```bash
mv data.jsonl ./test_data/
```

## Step 3: Custom Metric Configuration
In `./test/`, a file named `score.py` is already created to call **GPT-4o** to evaluate how closely the predicted answer matches the reference answer on a **0-10** scale.

To enable the usage of GPT-4o, set your OpenAI API key as follows:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk_xxxxxxxx"
```

## Step 4: Configuration Update

1. Test Environment Configuration

Modify `examples/llm_simple_qa/testenv/testenv.yaml`:

- Set the dataset path to point to `data.jsonl`.
- Change the metric field to `score` and update any necessary URLs.

2. Benchmarking Job Configuration

Modify `examples/llm_simple_qa/benchmarkingjob.yaml`:

- Specify the correct `workspace`, `testenv`, and `algorithms` paths.
- Set `metrics` and `sort_by` to `score`.

## Step 5: Run the Benchmark

Execute the following command from the Ianvs root directory:

```bash
python benchmarking.py -f examples/llm_simple_qa/benchmarkingjob.yaml
```

### Sample Output

```
(ianvs38) ➜  ianvs git:(main) ✗ python benchmarking.py -f examples/llm_simple_qa/benchmarkingjob.yaml
BaseModel doesn't need to train
BaseModel doesn't need to save
BaseModel load
BaseModel predict
+------+-------------------------------+-------+--------------------+-----------+---------------------+-----------------------------------------------------------------------------------------------------------------------+
| rank |           algorithm           | score |      paradigm      | basemodel |         time        |                                                          url                                                          |
+------+-------------------------------+-------+--------------------+-----------+---------------------+-----------------------------------------------------------------------------------------------------------------------+
|  1   | simple_qa_singletask_learning |  6.12 | singletasklearning |    gen    | 2025-02-23 14:06:34 | ./examples/llm_simple_qa/workspace/benchmarkingjob/simple_qa_singletask_learning/3a65ad0a-f1ac-11ef-b930-d59c724d9192 |
+------+-------------------------------+-------+--------------------+-----------+---------------------+-----------------------------------------------------------------------------------------------------------------------+
```

By following these steps, you can configure an edge-based environment using **KubeEdge-Ianvs**, leverage a specialized financial QA dataset refined by **GPT-4o**, and implement a custom **GPT-4o** scoring metric. This workflow ensures that your algorithm is tested in a setting that emphasizes **data privacy, real-time analysis, and domain-specific accuracy**—key elements in modern financial edge computing.
