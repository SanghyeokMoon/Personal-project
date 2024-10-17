import math

class Polynomial():
    def __init__(self, dim, lr=1e-5):
        self.dim = dim
        self.weights = [rand() * 0.001 for i in range(self.dim)] # initialization with a list type
        self.bias = 2.5 # initialization
        self.lr = lr # learning rate
        
    def forward(self, x):
        # To compute the weighted sum of Polynomial regression model
        # Bias 식도 포함해야 합니다.
        prediction = [self.weight[i] * x[i] + self.bias for i in range(len(x))]    ############################
        return prediction
        
    def backward(self, x, y):
        # To compute the prediction error (derivative of L=1/2 * (prediction - y)^2 by prediction)
        pred = self.forward(x) 
        errors = pred - y
        return errors
        
    def train(self, x, y, epochs):
        for e in range(epochs): # epochs 만큼 학습
            for i in range(len(y)): # 데이터 하나씩 학습
                x_, y_ = x[i], y[i] # Each data point
                
                # To update the weights and bias with backward() 
                errors = self.backward(x_, y_)
                # 각 차수의 weights update
                for j in range(len(self.weights)):
                    gradient_weights = errors * _x[j] ############################
                    self.weights[j] -=  gradient_weights * self.lr
                # bias update
                gradient_bias = errors * 1 ############################
                self.bias -= gradient_bias * self.lr
                
    def evaluate(self, x):
        # To compute the predictions with forward()
        predictions = [self.forward(x_) for x_ in x]
        return predictions # list type
    

# Model define and training

# Define a model
polynomial = Polynomial(dim=2, lr=1e-6)  #  위에서 구현한 Polynomial regression model 모델 정의

# Training
polynomial.train(X_train, y_train, 1000)   #  100 epoch 학습

# Print weight and bias
for i, weight in enumerate(polynomial.weights):
    print(f"weight_{i+1}: {weight:0.6f}")
print(f"bias: {polynomial.bias:0.6f}")