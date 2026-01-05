# How to Run

## Quick Start 

For a quick start , run :

```bash
chmod +x run.sh 
./run.sh
```

## Manual Setup 

### Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Code 

#### Generate datasets
```bash
python3 code/gen_data.py
```

#### Train Part 1 (Math GPT)
```bash
python3 code/train_part1.py
```

#### 3. Train Part 2 (Boolean GPT)
```bash
python3 code/train_part2.py
```

#### 4. Run demonstrations
```bash
python3 main.py
```

This is a submission for ML Final Assignment
