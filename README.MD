# SET UP

### Windows Requirements
- Python 3.10.11
- Tensorflow GPU 2.10.0
- Numpy 1.26.4
- CUDA 11.2
- CuDNN 8.1

### Run
```
cd /d/"INTELLIGENT"
source ojam_310_env/Scripts/activate
```

### Check Tensorflow GPU Avail
```
import tensorflow as tf

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(gpus)}")
    if gpus:
        print("TensorFlow GPU check PASSED")
        try:
            for gpu in gpus:
                print(f"  Name: {gpu.name}")
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set successfully.")
        except RuntimeError as e:
            print(f"RuntimeError setting memory growth: {e}")
        return True
    else:
        print("TensorFlow GPU check FAILED")
        return False

if __name__ == "__main__":
    check_gpu()
```

#### Not Avail
- Uninstall all related with NVIDIA, include the driver
- Install CUDA 11.2
- Install Supported NVIDIA Driver based on your specification device

### Terminate
```
deactivate
```

### Environtment
```
python -m venv ojam_310_env
```

### Kernel
```
pip install ipykernel

python -m ipykernel install --user --name=ojam_310_env --display-name "310 (GPU)"
```

#### Uninstall
```
jupyter kernelspec list
jupyter kernelspec uninstall "ojam_310_env"
```

### Library
- Tensorflow GPU
- Pandas
- Jupyter
- Torch
- Matplotlib
- NLTK
- Scikit-Learn
- Scipy
- Google Play Scraper
- Transformers
- ipykernel