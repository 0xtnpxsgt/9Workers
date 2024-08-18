# 9Workers 
 
- You must need to buy a VPS for running Allora Worker
- You can buy from : Contabo
- You should buy VPS which is fulfilling all these requirements : 
```bash
Operating System : Ubuntu 22.04
CPU: Minimum of 1/2 core.
Memory: 2 to 4 GB.
Storage: SSD or NVMe with at least 5GB of space.
```
# Install dependecies
### Deployment - Read Carefully! 
## Step 1: 
```bash
# Install Packages
sudo apt update & sudo apt upgrade -y

sudo apt install ca-certificates zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev curl git wget make jq build-essential pkg-config lsb-release libssl-dev libreadline-dev libffi-dev gcc screen unzip lz4 -y
```

## Step 2: 
```bash
# Install Python3
sudo apt install python3
python3 --version

sudo apt install python3-pip
pip3 --version
```

## Step 3: 
```bash
# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
docker version

# Install Docker-Compose
VER=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d '"' -f 4)

curl -L "https://github.com/docker/compose/releases/download/"$VER"/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

chmod +x /usr/local/bin/docker-compose
docker-compose --version

# Docker Permission to user
sudo groupadd docker
sudo usermod -aG docker $USER
```

## Step 4: 
```bash
# Install Go
sudo rm -rf /usr/local/go
curl -L https://go.dev/dl/go1.22.4.linux-amd64.tar.gz | sudo tar -xzf - -C /usr/local
echo 'export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin' >> $HOME/.bash_profile
echo 'export PATH=$PATH:$(go env GOPATH)/bin' >> $HOME/.bash_profile
source .bash_profile
go version
```


## Step 4: Install & RUN
```bash
cd $HOME && cd basic-coin-prediction-node
docker compose down -v
docker container prune

cd $HOME && rm -rf basic-coin-prediction-node
```



```bash
git clone https://github.com/allora-network/allora-huggingface-walkthrough
cd allora-huggingface-walkthrough
```
```bash
mkdir -p worker-data
chmod -R 777 worker-data
```
```bash
cp config.example.json config.json
nano config.json
```

####  Edit addressKeyName & addressRestoreMnemonic / Copy & Paste Inside config.json
#### Optional: RPC :  https://allora-rpc.testnet-1.testnet.allora.network/
```bash
{
    "wallet": {
        "addressKeyName": "test",
        "addressRestoreMnemonic": "your phase",
        "alloraHomeDir": "/root/.allorad",
        "gas": "1000000",
        "gasAdjustment": 1.0,
        "nodeRpc": "https://sentries-rpc.testnet-1.testnet.allora.network/",
        "maxRetries": 1,
        "delay": 1,
        "submitTx": false
    },
    "worker": [
        {
            "topicId": 1,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 1,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "ETH"
            }
        },
        {
            "topicId": 2,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 3,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "ETH"
            }
        },
        {
            "topicId": 3,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 5,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "BTC"
            }
        },
        {
            "topicId": 4,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 2,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "BTC"
            }
        },
        {
            "topicId": 5,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 4,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "SOL"
            }
        },
        {
            "topicId": 6,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 5,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "SOL"
            }
        },
        {
            "topicId": 7,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 2,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "ETH"
            }
        },
        {
            "topicId": 8,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 3,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "BNB"
            }
        },
        {
            "topicId": 9,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 5,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "ARB"
            }
        }
        
    ]
}
```
## Export 
```bash
chmod +x init.config
./init.config
```
## Edit App.py
- Register on Coingecko https://www.coingecko.com/en/developers/dashboard & Create Demo API KEY
- Copy & Replace API with your `COINGECKO API` , then save `Ctrl+X Y ENTER`.
```bash
nano app.py
```
```
from flask import Flask, Response
import requests
import json
import pandas as pd
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
import mxnet as mx
from gluonts.evaluation.backtest import make_evaluation_predictions
import traceback
import logging

# create our Flask app
app = Flask(__name__)

def get_coingecko_url(token):
    base_url = "https://api.coingecko.com/api/v3/coins/"
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum'
    }
    
    token = token.upper()
    if token in token_map:
        url = f"{base_url}{token_map[token]}/market_chart?vs_currency=usd&days=30&interval=daily"
        return url
    else:
        raise ValueError("Unsupported token")

@app.route("/inference/<string:token>")
def get_inference(token):
    """Generate inference for given token."""
    try:
        # get the data from Coingecko
        url = get_coingecko_url(token)
    except ValueError as e:
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')

    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "CG-XXXXXXXXXXXXXXXXXXXXXXXX" # replace with your API key
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["prices"])
        df.columns = ["date", "price"]
        df["date"] = pd.to_datetime(df["date"], unit='ms')
        df = df[:-1]  # removing today's price
    else:
        return Response(json.dumps({"Failed to retrieve data from the API": str(response.text)}), 
                        status=response.status_code, 
                        mimetype='application/json')

    # Preparing the data for DeepAR
    prediction_length = 1
    training_data = ListDataset([{"start": df["date"][0], "target": df["price"][:-prediction_length]}],
                                freq="D")

    try:
        # Define the DeepAR estimator
        estimator = DeepAREstimator(freq="D",
                                    prediction_length=prediction_length,
                                    trainer=Trainer(epochs=5))

        # Train the model and predict
        predictor = estimator.train(training_data=training_data)

        # Get predictions
        forecast_it, ts_it = make_evaluation_predictions(dataset=training_data, predictor=predictor)
        forecasts = list(forecast_it)
        ts = list(ts_it)

        # Access the mean of the forecasts
        forecast_mean = forecasts[0].mean_ts.item()

        return Response(str(forecast_mean), status=200)
    except Exception as e:
        logging.error(f"Inference error: {e}")
        traceback.print_exc()  # Print the full stack trace for debugging
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

# run our Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
```
## Edit requirements.txt
```bash
nano requirements.txt
```
```
flask[async]
gunicorn[gthread]
transformers[torch]
pandas
python-dotenv
gluonts>=0.11.4,<0.12.0
mxnet
numpy==1.19.5
orjson
```
## Build
```bash
docker compose up --build -d
```

## Check Logs 
```bash
docker compose logs -f worker
```

## Check your wallet here: http://worker-tx.nodium.xyz/
![image](https://github.com/user-attachments/assets/6e9ce7fd-fdf5-40d2-98f9-d20eb8486fce)




















