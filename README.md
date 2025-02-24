# Logistic Regression Neural Network
# Cat Image Classification using Logistic Regression Neural Network

## Overview
This project implements a logistic regression neural network to classify images as cats or non-cats. It's an assignment from the DeepLearning.AI course that demonstrates fundamental concepts of neural networks and deep learning.

## Prerequisites
* Python 3.7+
* NumPy
* Matplotlib
* h5py
* SciPy
* PIL (Python Imaging Library)
* Jupyter Notebook

Install required packages:
```bash
pip install numpy matplotlib h5py scipy pillow jupyter
```

## Dataset
* Training set: 209 labeled images
* Test set: 50 labeled images
* Image dimensions: 64x64x3 (RGB)
* Classes: Cat (1) and Non-cat (0)

## Project Structure
```
├── Logistic_Regression_with_a_Neural_Network_mindset-2.ipynb
└── README.md
```

## Implementation Details

### Data Preprocessing
* Flatten 64x64x3 images into 12288x1 vectors
* Normalize pixel values (divide by 255)
* Reshape data for computation

### Model Components
1. **Sigmoid Activation**
   * Converts linear output to probability
   * Range: [0,1]

2. **Parameter Initialization**
   * Weights (w): Zero vector
   * Bias (b): Zero scalar

3. **Forward Propagation**
   * Linear transformation: Z = wᵀX + b
   * Activation: A = sigmoid(Z)

4. **Backward Propagation**
   * Calculate gradients
   * Update parameters

5. **Optimization**
   * Gradient descent
   * Learning rate: 0.005
   * Iterations: 2000

## Running the Model

1. Open Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to and open `Logistic_Regression_with_a_Neural_Network_mindset-2.ipynb`

3. Run all cells sequentially

## Model Performance
* Training accuracy: ~99%
* Test accuracy: ~70%

## Hyperparameters
* Learning rate: 0.005
* Number of iterations: 2000
* Cost printing frequency: 100 iterations

## Troubleshooting

### Common Issues
1. **Memory Errors**
   * Reduce batch size
   * Process fewer images

2. **Poor Performance**
   * Adjust learning rate
   * Increase iterations
   * Check preprocessing

3. **Import Errors**
   * Verify package installation
   * Check Python version

## Contributing
* Fork the repository
* Submit pull requests
* Report issues
* Suggest improvements

## Acknowledgments
* DeepLearning.AI course materials
* Course instructors
* Deep learning community

## License
This project is part of the DeepLearning.AI coursework and follows their terms of use.

---

*Note: This implementation is for educational purposes and demonstrates basic neural network concepts.*
