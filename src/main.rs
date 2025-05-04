mod reader;
mod trainer;
mod tests;

use reader::{filter_0_and_2_as_binary, extract_ones};
use trainer::{train_and_evaluate, predict};
use std::path::Path;
use std::fs::File;
use std::io::{Write, BufWriter};

fn main() {
    let csv_path = "exoplanets_data.csv";
    if !Path::new(csv_path).exists() {
        panic!("File '{}' not found!", csv_path);
    }

    //trains the model on 0 and 2 labeled data
    let (features, label_enums) = filter_0_and_2_as_binary(csv_path);
    let labels: Vec<u8> = label_enums.iter().map(|&label| label.into()).collect();
    let report = train_and_evaluate(features, labels);

    println!("\nModel Evaluation:");
    println!("Accuracy: {:.2}%", report.accuracy * 100.0);
    println!("F1 Score: {:.4}", report.f1_score);

    //predicts for koi_disposition == 1
    let one_features = extract_ones(csv_path);
    let pred_classes = predict(&report.model, one_features);

    //outputs predictions to a text file
    let output_path = "exoplanet_predictions.txt";
    let file = File::create(output_path).expect("Failed to create prediction file");
    let mut writer = BufWriter::new(file);

    writeln!(writer, "Predictions for koi_disposition == 1").unwrap();
    writeln!(writer, "Format: PlanetIndex → Predicted Class").unwrap();

    for (i, &p) in pred_classes.iter().enumerate() {
        let label = match p {
            0 => "FalsePositive",
            1 => "Confirmed",
            _ => "Unknown",
        };
        writeln!(writer, "Planet {} → {}", i, label).unwrap();
    }
    println!("\nPredictions written to '{}'.", output_path);
}
