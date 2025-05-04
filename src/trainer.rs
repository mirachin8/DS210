// trains the decision tree model
use linfa::prelude::*;
use linfa::Dataset;
use linfa_trees::DecisionTree;
use linfa::metrics::ToConfusionMatrix;
use ndarray::{Array1, Array2};

//struct keeps my code organized and makes it easy to access accuracy and f1 score later
pub struct ModelReport {
    pub model: DecisionTree<f64, usize>,
    pub predictions: Array1<usize>,
    pub test_features: Array2<f64>,  // Store features separately
    pub test_labels: Array1<usize>,  // Store labels separately
    pub accuracy: f32,
    pub f1_score: f32,
}

//training function
pub fn train_and_evaluate(features: Array2<f64>, labels: Vec<u8>) -> ModelReport {
    let labels_usize: Array1<usize> = Array1::from_vec(labels.into_iter().map(|l| l as usize).collect());
    let dataset = Dataset::new(features, labels_usize);

    let (train, test) = dataset.split_with_ratio(0.8);
    let model = DecisionTree::params().fit(&train).unwrap();
    
    // Extract test features and labels
    let test_features = test.records().to_owned();
    let test_labels = test.targets().to_owned();
    
    let predictions = model.predict(&test);
    let cm = predictions.confusion_matrix(&test).unwrap();

    ModelReport {
        model,
        predictions,
        test_features,
        test_labels,
        accuracy: cm.accuracy(), //uses the confusionmatrix 
        f1_score: cm.f1_score(),
    }
}

//prediction function
pub fn predict(model: &DecisionTree<f64, usize>, x: Array2<f64>) -> Vec<usize> {
    model.predict(&x).into_raw_vec()
}

