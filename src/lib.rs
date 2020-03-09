// TODO: maybe change Vec<T> to 1array (need ndarray crate)

use num_traits::Float;

// Define generic Kernel trait as a struct with a .call(x1, x2) -> float method
pub trait Kernel<T, U: Float> {
    fn call(&self, x1: &T, x2: &T) -> U;
}

// RBF: Kernel with single float parameter: gamma
pub struct RBF<U: Float> {
    gamma: U,
}

// .call() for RBF is exp(-(x1 - x2)^2 / gamma)
impl<U: Float> Kernel<Vec<U>, U> for RBF<U> {
    fn call(&self, x1: &Vec<U>, x2: &Vec<U>) -> U {
        let mut neg_sq_dists: U = U::zero();
        for (x, y) in x1.iter().zip(x2.iter()) {
            neg_sq_dists = neg_sq_dists - (*x - *y).powi(2);
        }
        neg_sq_dists = neg_sq_dists / self.gamma;
        neg_sq_dists.exp()
    }
}

// primary SVM -- consists of a Kernel struct, a vector of alphas (initially None), and the dataset
//    used during training (initially None)
pub struct SVM<'a, T, U: Float> {
    kernel: Box<dyn Kernel<T, U>>,
    alpha: Option<Vec<U>>,
    bias: Option<U>,
    x_train: &'a Option<Vec<T>>,
    y_train: &'a Option<Vec<U>>
}

impl<'a, T, U: Float> SVM<'a, T, U>{
    pub fn new(k: Box<dyn Kernel<T, U>>) -> SVM<'a, T, U> {
        SVM {
            kernel: k,
            alpha: None,
            bias: None,
            x_train: &None,
            y_train: &None,
        }
    }

    pub fn set_alpha(&mut self, alpha_new: Vec<U>) {
        self.alpha = Some(alpha_new);
    }

    pub fn set_bias(&mut self, bias_new: U) {
        self.bias = Some(bias_new);
    }

    pub fn set_x_train(&mut self, x_train_new: &'a Option<Vec<T>>) {
        self.x_train = x_train_new;
    }

    pub fn set_y_train(&mut self, y_train_new: &'a Option<Vec<U>>) {
        self.y_train = y_train_new;
    }

    pub fn classify(&self, x: &T) -> Option<U> {
        if self.alpha.is_none() || self.bias.is_none() || self.x_train.is_none() {
            return None;
        } else {
            let mut sum_inner_prods = self.bias.unwrap();
            for (y, a) in self.x_train.as_ref().unwrap().iter().zip(self.alpha.as_ref().unwrap().iter()) {
                // this can be made parallel
                sum_inner_prods = sum_inner_prods + *a * self.kernel.call(x, y);
            }
            if sum_inner_prods > U::zero() {
                return U::from(1)
            } else{ 
                return U::from(-1)
            }
        }
    }

    pub fn regress(&self, x: &T) -> Option<U> {
        if self.alpha.is_none() || self.bias.is_none() || self.x_train.is_none() {
            return None;
        } else {
            let mut sum_inner_prods = self.bias.unwrap();
            for (y, a) in self.x_train.as_ref().unwrap().iter().zip(self.alpha.as_ref().unwrap().iter()) {
                // this can be made parallel
                sum_inner_prods = sum_inner_prods + *a * self.kernel.call(x, y);
            }
            return Some(sum_inner_prods);
        }
    }
}

fn smo_cls<'a, T, U: Float>(svm: &mut SVM<'a, T, U>, 
                            c: &U, 
                            x_train: &'a Option<Vec<T>>, 
                            y_train: &'a Option<Vec<U>>,
                            tol: &U,
                            max_iter: usize) {
    let xs = x_train.as_ref().unwrap();
    let ys = y_train.as_ref().unwrap();
    if x_train.is_some() && y_train.is_some() {
        svm.set_x_train(x_train);
        svm.set_y_train(y_train);
        let mut alpha = vec![U::zero(); xs.len()];
        let mut bias = U::zero();
        // while progress < tol and remaining iter != 0 do smo_step

        // after loop
        svm.set_alpha(alpha);
        svm.set_bias(bias);
    }
}

fn smo_step_cls<'a, T, U: Float>(svm: &mut SVM<'a, T, U>, c: &U) {

}