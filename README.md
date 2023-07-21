# Pending

**Dependencies**: Python 3.10.10

1. Need to add piece of code to move model directory from container to jfrog in train.py.
2. Need to add piece of code to copy model directory from jfrog to container in serve.py.

# Training

## Build Docker image
To build the Docker image for training, execute the following command:
```
docker build -t ilovepython/training:v1 -f Dockerfile.training .
```

## Run the training job in Kubernetes
Before deploying the training job, make sure to update the `training.yaml` file with the correct image tag `ilovepython/training:v1`. Then, use `kubectl` to deploy the training job in a specific namespace:
```
kubectl create -f training.yaml -n <namespace>
```
Replace `<namespace>` with the desired namespace where you want to deploy the training job.

# Serving

## Build Docker image
To build the Docker image for serving, run the following command:
```
docker build -t ilovepython/serving:v1 -f Dockerfile.serving .
```

## Run the serving job in Kubernetes
Before deploying the serving job, ensure that you update the `serving.yaml` file with the correct image tag `ilovepython/serving:v1`. Then, deploy the serving job using `kubectl` in the desired namespace:
```
kubectl create -f serving.yaml -n <namespace>
```
Replace `<namespace>` with the target namespace where you wish to deploy the serving job.

# Prediction

To make a prediction using the deployed serving service, use the following command:
```
curl -X POST -F "image_file=@sample_images/sample-2.png" http://127.0.0.1:8000/predict
```
Ensure that you have the correct image file path for the `sample-2.png` image file. This command will send the image to the serving endpoint, and you will receive the prediction results in the response.