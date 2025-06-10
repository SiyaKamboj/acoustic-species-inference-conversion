//1. PREPROCESSING

//A. parse the json files from dataset
//matching rust constructs to the separated code
use serde::Deserialize;
//load json file
use std::fs::File;
use std::io::BufReader;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyList, PyModule};
use serde::{Deserializer};
use serde::de::Error as DeError;
use serde::de::{self, DeserializeOwned};
use ndarray::Array1;
use ndarray::Array2;
use ndarray_npy::write_npy;
use std::fs;
use std::path::{Path, PathBuf};
use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor, GraphOptimizationLevel, LoggingLevel, session::Session};
use ndarray_npy::read_npy;
use ndarray::Array3;
use ndarray::Array4;
use ndarray::IxDyn;
use ndarray::Ix2;
use std::sync::Arc;
use ndarray::ArrayD;
use itertools::Itertools;
//use sysinfo::{System, SystemExt, CpuExt};
use sysinfo::System;
use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Local; // Add to Cargo.toml: chrono = "0.4"

//create structs to match the json files - a requirement for parsing with serde in Rust
#[derive(Debug, Deserialize)]
#[derive(Default)] //if it is missing, fills it to default values
struct Audio {
    path: String,
    #[serde(default)]
    bytes: Option<serde_json::Value>,  // null in the dataset
}
#[derive(Debug, Deserialize)]
struct Sample {
    audio: Audio,
    filepath: String,
    //start_time: f32,
    #[serde(default)]
    start_time: Option<f32>,
    //end_time: f32,
    #[serde(default)]
    end_time: Option<f32>,
    
    #[serde(default)]
    low_freq: Option<f32>,
    #[serde(default)]
    high_freq: Option<f32>,
    #[serde(default, deserialize_with = "int_to_string")]
    ebird_code: Option<String>,
    #[serde(default)]
    ebird_code_multilabel: Vec<u32>,
    #[serde(default, deserialize_with = "null_to_empty_vec")]
    ebird_code_secondary: Vec<String>,
    #[serde(default)]
    call_type: Option<String>,
    #[serde(default)]
    sex: Option<String>,
    #[serde(default)]
    lat: Option<f64>,
    #[serde(default)]
    long: Option<f64>,
    #[serde(default)]
    length: Option<f32>,
    #[serde(default)]
    microphone: Option<String>,
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    local_time: Option<String>,
    #[serde(default)]
    detected_events: Option<serde_json::Value>,
    #[serde(default)]
    event_cluster: Option<serde_json::Value>,
    #[serde(default)]
    peaks: Option<serde_json::Value>,
    #[serde(default)]
    quality: Option<String>,
    #[serde(default)]
    recordist: Option<String>,
    //#[serde(default)]
    #[serde(default, deserialize_with = "int_to_string")]
    genus: Option<String>,
    //#[serde(default)]
    #[serde(default, deserialize_with = "int_to_string")]
    species_group: Option<String>,
    //#[serde(default)]
    #[serde(default, deserialize_with = "int_to_string")]
    order: Option<String>,
    #[serde(default)]
    genus_multilabel: Vec<u32>,
    #[serde(default)]
    species_group_multilabel: Vec<u32>,
    #[serde(default)]
    order_multilabel: Vec<u32>,
    #[serde(default)]
    audio_in: Audio,
    #[serde(default)]
    labels: Vec<u32>,
}

// Custom deserializer to handle both string and number types for fields like ebird_code
fn int_to_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let val: serde_json::Value = Deserialize::deserialize(deserializer)?;
    match val {
        serde_json::Value::String(s) => Ok(Some(s)),
        serde_json::Value::Number(n) => Ok(Some(n.to_string())),
        serde_json::Value::Null => Ok(None),
        _ => Err(D::Error::custom("expected string or number")),
    }
}
fn null_to_empty_vec<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    let opt = Option::<Vec<T>>::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

// Function to load a JSON split file and parse it into a vector of Sample structs
fn load_split(path: &str) -> Result<Vec<Sample>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let data: Vec<Sample> = serde_json::from_reader(reader)?;
    Ok(data)
}

//Function to make sure that the split works
fn run_json_processing() -> Result<(), Box<dyn std::error::Error>> {
    let test_data = load_split("../birdset_export/HSN/test.json")?;
    println!("First test sample path: {}", test_data[0].audio.path);
    Ok(())
}


//Convert everything into spectograms and save them as .npy files, so that it can be used as an input for the onnx model
fn run_python_preprocessing(split_data: &Vec<Sample>, split_name: &str) -> PyResult<()> {
    Python::with_gil(|py| {
        let wrapper = PyModule::from_code(
            py,
            &std::fs::read_to_string("preprocessor_wrapper.py").unwrap(),
            "preprocessor_wrapper.py",
            "preprocessor_wrapper",
        )?;

        let label_names = vec![
            "gcrfin", "whcspa", "amepip", "sposan", "rocwre", "brebla", "daejun", "foxspa",
            "clanut", "moublu", "casfin", "mallar3", "herthr", "amerob", "yerwar", "yelwar",
            "dusfly", "mouchi", "orcwar", "warvir", "norfli",
        ];

        wrapper.getattr("init_preprocessor")?.call1((label_names, 5))?;

        let out_dir = Path::new("../mel_outputs");
        fs::create_dir_all(out_dir)?; // create mel_outputs if it doesn't exist

        for sample in split_data.iter() {
            let path = &sample.audio.path;
            let labels = &sample.labels;

            // Get filename stem from path
            let file_stem = Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("sample_unknown");
            
            // Define output paths
            let mel_path= out_dir.join(format!("{}/{}_mel.npy", split_name, file_stem));
            let label_path = out_dir.join(format!("{}/{}_label.npy", split_name, file_stem));

            // Skip if already processed
            if mel_path.exists() && label_path.exists() {
                println!("Skipping already processed: {}", file_stem);
                continue;
            }

            let py_labels = PyList::new(py, labels.clone());

            let result = wrapper
                .getattr("process_audio_file")?
                .call1((path, py_labels))?
                //.extract::<&PyDict>()?;
                .extract::<&pyo3::types::PyDict>()?;

            let mel = result.get_item("mel").unwrap();
            let one_hot = result.get_item("labels").unwrap();
            
            let mel_val = mel.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing mel value"))?;
            // Extract as a 3D Vec
            let mel_vec: Vec<Vec<Vec<f32>>> = mel_val.extract()?;
            // Get dimensions
            let (d0, d1, d2) = (
                mel_vec.len(),
                mel_vec[0].len(),
                mel_vec[0][0].len()
            );
            // Flatten into 1D Vec
            let flattened: Vec<f32> = mel_vec
                .into_iter()
                .flatten()
                .flatten()
                .collect();

            // Convert to ndarray
            let mel_arr = ndarray::Array3::from_shape_vec((d0, d1, d2), flattened).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Shape error: {}", e)))?;;
            // Save as .npy
            ndarray_npy::write_npy(mel_path, &mel_arr).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to save mel: {}", e)))?;

            // Save one-hot
            let label_val = one_hot.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing label value"))?;
            let label_vec: Vec<f32> = label_val.extract()?;
            let label_array = Array1::from(label_vec);
            ndarray_npy::write_npy(label_path, &label_array).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to save label: {}", e)))?;;

            println!("Processed and saved: {}", file_stem);
        }

        Ok(())
    })
}


fn run_inference() -> Result<(), Box<dyn std::error::Error>> {
    let num_epochs = 6;
    // Load ONNX model
    let environment = Arc::new(Environment::builder().with_name("inference").build()?);
    let mut session = environment
        .new_session_builder()?
        .with_model_from_file("efficientnet_b1_export.onnx")?;

    // Label names
    let label_names = vec![
        "gcrfin", "whcspa", "amepip", "sposan", "rocwre", "brebla", "daejun", "foxspa",
        "clanut", "moublu", "casfin", "mallar3", "herthr", "amerob", "yerwar", "yelwar",
        "dusfly", "mouchi", "orcwar", "warvir", "norfli"
    ];

    //Run inference on mel files in the test directory
    let mel_dir = Path::new("../mel_outputs/test");

    for epoch in 1..=num_epochs {
        println!("Epoch {}", epoch);
        let epochStart = Instant::now();
        for entry in fs::read_dir(mel_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|ext| ext == "npy").unwrap_or(false)
                && path.file_name().unwrap_or_default().to_str().unwrap_or("").contains("_mel")
            {
                let mel: ArrayD<f32> = read_npy(&path)?;

                // Reshape to [1, 1, 256, 431]
                let mel = mel.into_dimensionality::<ndarray::Ix3>()?;
                let mel = mel.insert_axis(ndarray::Axis(0));
                let input_tensor: Vec<ArrayD<f32>> = vec![mel.into_dyn()];

                // Run inference
                let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor)?;

                // Get output
                let output_tensor = &outputs[0];
                let logits = output_tensor
                    .view()
                    .to_owned()
                    .into_dimensionality::<Ix2>()?;
                let logits_row = logits.row(0);

                // Get top predicted class index and score
                if let Some((idx, score)) = logits_row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                    // println!(
                    //     "File: {} → Top Prediction: Class {} ({}) with score {:.4}",
                    //     path.file_name().unwrap().to_string_lossy(),
                    //     idx,
                    //     label_names[idx],
                    //     score
                    // );
                } else {
                    println!(
                        "File: {} → No prediction could be made.",
                        path.file_name().unwrap().to_string_lossy()
                    );
                }
            }
        }
        let epochduration = epochStart.elapsed();
        println!("Epoch {} inference time: {:.2?}", epoch, epochduration);
    }

    Ok(())
}


fn load_mel_files_from_split(split: &str) -> Result<Vec<Array3<f32>>, Box<dyn std::error::Error>> {
    let dir_path = format!("../mel_outputs/{}/", split);
    let mut mel_arrays = Vec::new();

    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("npy")
            && path.file_name().unwrap().to_string_lossy().contains("_mel")
        {
            let mel_array: Array3<f32> = read_npy(path)?;
            mel_arrays.push(mel_array);
        }
    }

    Ok(mel_arrays)
}



fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_json_processing()?; // Run your Rust JSON logic

    //let start = Instant::now();
    log_system_usage("Before Preprocessing");
    //works!!!

    //run python preprocessing for multiple audio files
    let train_data = load_split("../birdset_export/HSN/train.json")?;
    let valid_data = load_split("../birdset_export/HSN/valid.json")?;
    let test_data = load_split("../birdset_export/HSN/test.json")?;

    match run_python_preprocessing(&train_data, "train") {
        Ok(_) => println!("Python preprocessing succeeded for train."),
        Err(e) => eprintln!("Python preprocessing failed for train: {:?}", e),
    }

    match run_python_preprocessing(&valid_data, "valid") {
        Ok(_) => println!("Python preprocessing succeeded for valid."),
        Err(e) => eprintln!("Python preprocessing failed for valid: {:?}", e),
    }

    match run_python_preprocessing(&test_data, "test") {
        Ok(_) => println!("Python preprocessing succeeded for test."),
        Err(e) => eprintln!("Python preprocessing failed for test: {:?}", e),
    }

    log_system_usage("After Preprocessing / Before Inference");
    //println!("Duration of preprocessing/before inference: {:?}", start.elapsed());

    // Run inference
    run_inference()?;
    println!("Inference completed successfully.");

    log_system_usage("After Inference");
    //println!("Duration after inference: {:?}", start.elapsed());
    Ok(())
}


// Log metrics and system usage
fn log_system_usage(phase: &str) {
    let mut sys = System::new_all();
    sys.refresh_all();

    let now = Local::now();
    let log = format!(
        "\n[{}] === System Usage during: {} ===\n\
        Total Memory: {} MB\n\
        Used Memory:  {} MB\n\
        CPU Usage:    {:.2}%\n\
        ======================================\n",
        now.format("%Y-%m-%d %H:%M:%S"),
        phase,
        sys.total_memory() / 1024,
        sys.used_memory() / 1024,
        sys.global_cpu_info().cpu_usage()
    );

    // Append to file
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("system_metrics.txt")
        .expect("Unable to open or create system_metrics.txt");

    writeln!(file, "{}", log).expect("Failed to write to system_metrics.txt");
}