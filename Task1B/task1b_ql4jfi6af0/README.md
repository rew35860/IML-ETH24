# 1A. Task description
# Linear Regression Task

## **Overview**
This project involves implementing **linear regression** to predict an output value **y** based on a set of **feature transformations** applied to an input vector **x**. The model uses **21 feature transformations**, including **linear, quadratic, exponential, cosine, and a constant term**.

## **Feature Transformations**
The input vector **x** consists of five components **x₁, x₂, x₃, x₄, x₅**. The following transformations are applied:

### **1. Linear Features**
```math
\phi_1(x) = x_1, \phi_2(x) = x_2, \phi_3(x) = x_3, \phi_4(x) = x_4, \phi_5(x) = x_5
```

### **2. Quadratic Features**
```math
\phi_6(x) = x_1^2, \phi_7(x) = x_2^2, \phi_8(x) = x_3^2, \phi_9(x) = x_4^2, \phi_{10}(x) = x_5^2 
```

### **3. Exponential Features**
```math
\phi_{11}(x) = e^{x_1}, \phi_{12}(x) = e^{x_2}, \phi_{13}(x) = e^{x_3}, \phi_{14}(x) = e^{x_4}, \phi_{15}(x) = e^{x_5}
```

### **4. Cosine Features**
```math
\phi_{16}(x) = \cos(x_1), \phi_{17}(x) = \cos(x_2), \phi_{18}(x) = \cos(x_3), \phi_{19}(x) = \cos(x_4), \phi_{20}(x) = \cos(x_5) 
```

### **5. Constant Term**
```math
\phi_{21}(x) = 1 
```

## **Model Formula**
The predicted value \( \hat{y} \) is given by:
```math
\hat{y} = w_1 \phi_1(x) + w_2 \phi_2(x) + \dots + w_{21} \phi_{21}(x)
```
where \( w_i \) are the model parameters (weights) to be learned from the data.

We provide a template solution file that suggests a structure for how you can solve the task, by filing in the TODOs in the skeleton code. It is not mandatory to use this solution template but it is recommended since it should make getting started on the task easier. You are also encouraged (but not required) to implement regression solutions from scratch, for a deeper understanding of the course material.



