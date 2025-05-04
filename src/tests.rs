//module for all tests
#[cfg(test)]
mod tests {
    use crate::reader::{filter_0_and_2_as_binary, extract_ones};
    use crate::trainer::{train_and_evaluate, predict};

    #[test]
    fn test_data_loading() {
        //tests the data loading and filtering functions
        let (features, labels) = filter_0_and_2_as_binary("exoplanets_data.csv");
        
        //basic dataset feature checks
        assert!(features.nrows() > 0, "Should load some data rows");
        assert_eq!(features.ncols(), 9, "Should have 9 feature columns");
        assert_eq!(features.nrows(), labels.len(), "Features and labels should have same number of rows");
        
        //checks that labels are binary (0 or 1)
        for label in &labels {
            assert!(*label == 0 || *label == 1, "Labels should be either 0 or 1");
        }
        
        //tests extract_ones
        let one_features = extract_ones("exoplanets_data.csv");
        assert!(one_features.nrows() > 0, "Should find some class 1 examples");
        assert_eq!(one_features.ncols(), 9, "Should have 9 feature columns");
    }

    #[test]
    fn test_model_pipeline() {
        //gets training data
        let (features, label_enums) = filter_0_and_2_as_binary("exoplanets_data.csv");
        let labels: Vec<u8> = label_enums.iter().map(|&label| label.into()).collect();
        
        //trains model
        let report = train_and_evaluate(features, labels);
        
        //checks if model metrics are valid
        assert!(report.accuracy >= 0.0 && report.accuracy <= 1.0, 
                "Accuracy should be between 0 and 1");
        
        //tests prediction functionality
        let one_features = extract_ones("exoplanets_data.csv");
        if one_features.nrows() > 0 {
            let predictions = predict(&report.model, one_features);
            
            //checks predictions
            assert!(!predictions.is_empty(), "Should produce predictions");
            
            //verifies all predictions are either 0 or 1
            for pred in predictions {
                assert!(pred == 0 || pred == 1, "Predictions should be binary (0 or 1)");
            }
        }
    }
}