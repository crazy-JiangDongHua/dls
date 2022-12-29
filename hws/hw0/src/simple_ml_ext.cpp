#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t batch_nums = ceil(1.0*m/batch);
    for(size_t batch_num=0;batch_num<batch_nums;batch_num++){
        const float *x_b = X+batch_num*batch*n;
        const unsigned char *y_b = y+batch_num*batch;
        size_t real_batch = (batch_num+1)*batch<=m?batch:m-batch_num*batch;
        // z_exp = np.exp(np.dot(x_b, theta))
        float z_exp[real_batch*k]={0.0};
        for(size_t i=0;i<real_batch;i++){
            for(size_t j=0;j<k;j++){
                for(size_t e=0;e<n;e++){
                    z_exp[i*k+j]+=x_b[i*n+e]*theta[e*k+j];
                }
                z_exp[i*k+j]=exp(z_exp[i*k+j]);
            }
        }
        // z_norm = z_exp/z_exp.sum(axis=1, keepdims=True)
        for(size_t i=0;i<real_batch;i++){
            float sum=0.0;
            for(size_t j=0;j<k;j++){
                sum+=z_exp[i*k+j];    
            }
            for(size_t j=0;j<k;j++){
                z_exp[i*k+j]/=sum;    
            }
        }
        // iy = np.zeros(z_exp.shape, dtype=np.float32)
        float iy[real_batch*k]={0.0};
        // iy[np.arange(x_b.shape[0]), y_b]=1.0
        for(size_t i=0;i<real_batch;i++){
            iy[i*k+y_b[i]]=1.0;
        }
        // grad = np.dot(x_b.T, z_norm-iy) / x_b.shape[0]
        float grad[n*k]={0.0};
        for(size_t i=0;i<n;i++){
            for(size_t j=0;j<k;j++){
                for(size_t e=0;e<real_batch;e++){
                    grad[i*k+j]+=x_b[e*n+i]*(z_exp[e*k+j]-iy[e*k+j]);
                }
                grad[i*k+j]/=real_batch;
            }
        }
        // theta[:,:] -= lr * grad
        for(size_t i=0;i<n;i++){
            for(size_t j=0;j<k;j++){
                theta[i*k+j]-=lr*grad[i*k+j];
            }
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
