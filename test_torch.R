# install.packages("torch")

library(torch)
# torch::install_torch()

num_feature <- 5L
w_actual <- matrix(sample(seq(-1, 1, by = 0.1), num_feature), ncol = 1L)
w_actual <- torch_tensor(w_actual)
b_actual <- torch_tensor(0.1)
x <- torch_randn(100, num_feature)
y <- b_actual + torch_matmul(x, w_actual)

w <- torch_randn(num_feature, 1, requires_grad = TRUE)
b <- torch_zeros(1, requires_grad = TRUE)

lr <- 0.5
for (i in 1:100) {
  y_hat <- torch_mm(x, w) + b
  loss <- torch_mean((y - y_hat$squeeze(1))^2)
  
  loss$backward()
  
  with_no_grad({
    w$sub_(w$grad*lr)
    b$sub_(b$grad*lr)   
    
    w$grad$zero_()
    b$grad$zero_()
  })
}
print(w)
print(b)
cat("Error in weights: ", as_array(w - w_actual)[, 1])
cat("Error in const: ", as_array(b - b_actual))
