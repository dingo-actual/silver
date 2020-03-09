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

// struct to cache and retrieve errors
struct ErrorCache<U: Float> {
    errors: Vec<U>,
    unset: Vec<bool>,
    last_alpha: Vec<U>
}

impl<U: Float> ErrorCache<U> {
    pub fn new(alphas: &Vec<U>) -> ErrorCache<U> {
        ErrorCache {
            errors: vec![U::zero(); alphas.len()],
            unset: vec![true; alphas.len()],
            last_alpha: alphas.clone()
        }
    }
    pub fn retrieve(&self, ix: &usize, alpha: &U) -> Option<U> {
        if self.unset[*ix] || (alpha != self.last_alpha[*ix]) {
            return None;
        } else {
            return Some(self.errors[*ix]);
        }
    }
    pub fn store(&mut self, ix: &usize, alpha: &U, error: &U) {
        self.unset[*ix] = false;
        self.last_alpha[*ix] = *alpha;
        self.errors[*ix] = *error;
    }
}


// primary SVM -- consists of a Kernel struct, a vector of alphas (initially None), and the dataset
//    used during training (initially None)
pub struct SVM<'a, T, U: Float> {
    kernel: Box<dyn Kernel<T, U>>,
    alpha: Vec<U>,
    bias: U,
    x_train: &'a Vec<T>,
    y_train: &'a Vec<U>
}

impl<'a, T, U: Float> SVM<'a, T, U>{
    pub fn new(k: Box<dyn Kernel<T, U>>, x_train: &'a Vec<T>, y_train: &'a Vec<U>) -> SVM<'a, T, U> {
        SVM {
            kernel: k,
            alpha: vec![U::zero(); x_train.len()],
            bias: U::zero(),
            x_train: x_train,
            y_train: y_train,
        }
    }

    pub fn set_alpha(&mut self, alpha_new: Vec<U>) {
        self.alpha = alpha_new;
    }

    pub fn set_bias(&mut self, bias_new: U) {
        self.bias = bias_new;
    }

    pub fn classify(&self, x: &T) -> U {
        let mut sum_inner_prods = self.bias;
        for (y, a) in self.x_train.iter().zip(self.alpha.iter()) {
            // this can be made parallel
            sum_inner_prods = sum_inner_prods + *a * self.kernel.call(x, y);
        }
        if sum_inner_prods > U::zero() {
            return U::from(1).unwrap()
        } else{ 
            return U::from(-1).unwrap()
        }
    }

    pub fn regress(&self, x: &T) -> U {
        let mut sum_inner_prods = self.bias;
        for (y, a) in self.x_train.iter().zip(self.alpha.iter()) {
            // this can be made parallel
            sum_inner_prods = sum_inner_prods + *a * self.kernel.call(x, y);
        }
        return sum_inner_prods;
    }
}

fn smo_cls<'a, T, U: Float>(svm: &mut SVM<'a, T, U>, 
                            c: &U, 
                            tol: &U,
                            max_iter: usize) {
    
}


fn smo_step_cls<'a, T, U: Float>(svm: &mut SVM<'a, T, U>, c: &U, ix1: &usize, ix2: &usize) -> bool {
    let err_cache = ErrorCache::new(&svm.alpha);
    if *ix1 == *ix2 {
        return false;
    } else {
        let alpha1 = svm.alpha[*ix1];
        let alpha2 = svm.alpha[*ix2];
        let y1 = svm.y_train[*ix1];
        let y2 = svm.y_train[*ix2];
        let e1_try = err_cache.retrieve(ix1, &alpha1);
        let e1 = match e1_try {
            None => {
                let out = svm.regress(&svm.x_train[*ix1]) - y1;
                err_cache.store(ix1, &alpha1, &out);
                out
            },
            Some(x) => x
        };
        let s = y1 * y2;
        // compute L

        // compute H

    }
        
}
