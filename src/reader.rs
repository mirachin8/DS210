// module to read the csv and do some data cleaning
use csv::Reader;
use serde::Deserialize;
use ndarray::Array2;

/// Struct to deserialize each CSV row into and define datatypes
#[derive(Debug, Deserialize)]
pub struct ExoplanetRow {
    pub koi_disposition: u8,
    pub koi_score: Option<f64>,
    pub koi_period: Option<f64>,
    pub koi_prad: Option<f64>,
    pub koi_teq: Option<f64>,
    pub koi_insol: Option<f64>,
    pub koi_steff: Option<f64>,
    pub koi_srad: Option<f64>,
    pub koi_slogg: Option<f64>,
    pub koi_kepmag: Option<f64>,
}

// Filters for koi_disposition == 0 or 2, converts 2 → 1
pub fn filter_0_and_2_as_binary(path: &str) -> (Array2<f64>, Vec<u8>) {
    let mut rdr = Reader::from_path(path).expect("Failed to open CSV file");
    let mut features = vec![];
    let mut labels = vec![];

    for result in rdr.deserialize() {
        let row: ExoplanetRow = result.expect("Malformed row");

        match row.koi_disposition {
            0 | 2 => {
                // Converts all values of 2 to 1, and keeps 0 as is
                let label = if row.koi_disposition == 2 { 1 } else { 0 };
                labels.push(label);

                let row_vec = vec![
                    row.koi_score.unwrap_or(0.0),
                    row.koi_period.unwrap_or(0.0),
                    row.koi_prad.unwrap_or(0.0),
                    row.koi_teq.unwrap_or(0.0),
                    row.koi_insol.unwrap_or(0.0),
                    row.koi_steff.unwrap_or(0.0),
                    row.koi_srad.unwrap_or(0.0),
                    row.koi_slogg.unwrap_or(0.0),
                    row.koi_kepmag.unwrap_or(0.0),
                ];
                features.push(row_vec); //new vector 2s as 1
            }
            _ => continue,
        }
    }

    let row_len = features.first().map(|r| r.len()).unwrap_or(0); //makes sure that there are no missing values
    assert!(
        features.iter().all(|r| r.len() == row_len),
        "Inconsistent row lengths in feature data"
    );

    let num_rows = features.len();
    let flat: Vec<f64> = features.into_iter().flatten().collect();
    let feature_matrix = Array2::from_shape_vec((num_rows, row_len), flat)
        .expect("Array shape error");
    (feature_matrix, labels)
}

// Extract rows where koi_disposition is 1
pub fn extract_ones(path: &str) -> Array2<f64> {
    let mut rdr = Reader::from_path(path).expect("Failed to open CSV file");
    let mut features = vec![];

    for result in rdr.deserialize() {
        let row: ExoplanetRow = result.expect("Malformed row");

        if row.koi_disposition == 1 { //the extraction step
            let row_vec = vec![
                row.koi_score.unwrap_or(0.0),
                row.koi_period.unwrap_or(0.0),
                row.koi_prad.unwrap_or(0.0),
                row.koi_teq.unwrap_or(0.0),
                row.koi_insol.unwrap_or(0.0),
                row.koi_steff.unwrap_or(0.0),
                row.koi_srad.unwrap_or(0.0),
                row.koi_slogg.unwrap_or(0.0),
                row.koi_kepmag.unwrap_or(0.0),
            ];
            features.push(row_vec);
        }
    }

    let row_len = features.first().map(|r| r.len()).unwrap_or(0);
    assert!(
        features.iter().all(|r| r.len() == row_len),
        "Inconsistent row lengths in feature data"
    );

    let num_rows = features.len();
    let flat: Vec<f64> = features.into_iter().flatten().collect();
    let feature_matrix = Array2::from_shape_vec((num_rows, row_len), flat)
        .expect("Array shape error");

    feature_matrix
}
