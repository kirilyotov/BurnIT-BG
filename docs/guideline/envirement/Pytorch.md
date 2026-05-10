# Install pytorch

1. Verify you have alrady installed cuda direvers
2. Verify that you are in your local venv
3. Install it from [here](https://pytorch.org/get-started/locally/) or if your nvidia supports latest cuda:
``` bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
``` 

4. Check if pytorch is working with cuda run [this](../../../experiments/local-envirement/test_pytorch.ipynb) Jupiter notebook. Note you should select kernel version (top right) before running it and select venv where you have alrady installed pytorch.