// trains the decision tree model
use linfa::prelude::*;
use linfa::Dataset;
use linfa_trees::DecisionTree;
use linfa::metrics::ToConfusionMatrix;
use ndarray::{Array1, Array2};