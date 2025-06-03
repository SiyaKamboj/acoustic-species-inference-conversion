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

#[derive(Debug, Deserialize)]
#[derive(Default)] //if it is missing, fills it to default values
//BUT THIS COULD CAUSE PROBLEMS IF THE JSON FILES ARE NOT COMPLETE SO THEN MAYBE LET IT TRHOW AN ERROR
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

fn load_split(path: &str) -> Result<Vec<Sample>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let data: Vec<Sample> = serde_json::from_reader(reader)?;
    Ok(data)
}

fn run_json_processing() -> Result<(), Box<dyn std::error::Error>> {
    let test_data = load_split("../birdset_export/HSN/test.json")?;

    println!("First test sample path: {}", test_data[0].audio.path);

    // let mut num = 0; // move this outside the loop if you want to count across all samples
    // for i in 0..test_data.len() {
    //     if test_data[i].labels.len() > 1 {
    //         num += 1;
    //         println!("Greater");
    //     }
    // }

    // println!("count of num is {}", num);
    Ok(())
}

fn run_python_preprocessing_one_audio_file() -> PyResult<()> {
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

        //works for one audio file
        let audio_path = "/Users/siyakamboj/.cache/huggingface/hub/datasets--DBD-research-group--BirdSet/snapshots/ee31c6ba7dd653e57bf327bdd0c1bde6b0334bba/HSN/extracted/HSN_test5s_shard_0001.tar.gz/HSN_093_20150712_085105_235_240.ogg";
        let label_ids = vec![0];

        let result = wrapper
            .getattr("process_audio_file")?
            .call1((audio_path, label_ids))?
            .extract::<&pyo3::types::PyDict>()?;

        let mel = result.get_item("mel").unwrap();
        let labels = result.get_item("labels").unwrap();

        println!("Mel shape: {:?}", mel);
        println!("One-hot labels: {:?}", labels);


        Ok(())
    })
}


//fn run_python_preprocessing(train_data: &Vec<Sample>) -> PyResult<()> {
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

        //for sample in train_data.iter() {
        for sample in split_data.iter() {
            let path = &sample.audio.path;
            let labels = &sample.labels;

            // Get filename stem from path
            let file_stem = Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("sample_unknown");
            
            // Define output paths
            // let mel_path = out_dir.join(format!("{}_mel.npy", file_stem));
            // let label_path = out_dir.join(format!("{}_label.npy", file_stem));
            let mel_path= out_dir.join(format!("{}/{}_mel.npy", split_name, file_stem));
            //let mel_path = format!("../mel_outputs/{}/{}_mel.npy", split_name, file_stem);
            //let label_path = format!("../mel_outputs/{}/{}_label.npy", split_name, file_stem);
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


            // // Save mel
            // //give error if mel is missing rather than panicking or returning None
            // let mel_val = mel.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing mel value"))?;
            // //let mel_vec: Vec<Vec<f32>> = mel.extract()?;
            // let mel_vec: Vec<Vec<f32>> = mel_val.extract()?;
            // // let mel_arr = ndarray::Array2::from_shape_vec(
            // //     (mel_vec.len(), mel_vec[0].len()),
            // //     mel_vec.into_iter().flatten().collect(),
            // // )?;
            // //write to mel array. any shape error becomes python error
            // let mel_arr = ndarray::Array2::from_shape_vec(
            //     (mel_vec.len(), mel_vec[0].len()),
            //     mel_vec.into_iter().flatten().collect(),
            // ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Shape error: {}", e)))?;
            // //ndarray_npy::write_npy(mel_path, &mel_arr)?;
            // ndarray_npy::write_npy(mel_path, &mel_arr).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to save mel: {}", e)))?;
            
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
            //let label_vec: Vec<u8> = one_hot.extract()?;
            //ndarray_npy::write_npy(label_path, &label_vec)?;

            println!("Processed and saved: {}", file_stem);
        }

        Ok(())
    })
}

// fn run_inference() -> Result<(), Box<dyn std::error::Error>> {
//     // Initialize ONNX Runtime environment
//     let environment = Environment::builder()
//         .with_name("pyha")
//         .with_log_level(LoggingLevel::Warning)
//         .build()?;

//     let mut session = environment
//         .new_session_builder()?
//         .with_model_from_file("efficientnet_b1_export.onnx")?; // Update with your path

//     // Load preprocessed mel input (e.g., ../mel_outputs/test/XC123_mel.npy)
//     let mel: ndarray::Array3<f32> = read_npy("mel_input.npy")?;

//     // Ensure the shape is (1, 256, 431) or (batch_size, height, width)
//     let input_tensor_values = vec![mel];

//     // Run inference
//     let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values)?;

//     // Print top prediction values
//     let logits = &outputs[0];
//     println!("Logits: {:?}", logits);

//     Ok(())
// }

// fn run_inference(mel: &Array3<f32>) -> Result<(), Box<dyn std::error::Error>> {
//     // Step 1: Add channel dimension -> shape: [1, 1, 256, 431]
//     let input_tensor: Array4<f32> = mel.clone().insert_axis(ndarray::Axis(1));

//     // Step 2: Create the runtime environment (you should do this once globally in production)
//     let environment = Arc::new(
//         Environment::builder()
//             .with_name("inference")
//             .with_log_level(LoggingLevel::Warning)
//             .build()?
//     );

//     // Step 3: Load the ONNX model
//     let session = environment
//         .new_session_builder()?
//         .with_optimization_level(GraphOptimizationLevel::Basic)?
//         .with_number_threads(1)?
//         .with_model_from_file("efficientnet_b1_export.onnx")?;

//     // Step 4: Run inference
//     let input_tensor_dyn = vec![input_tensor.into_dyn()];
//     let input_names = session.inputs.iter().map(|i| i.name.clone()).collect::<Vec<_>>();

//     //let outputs: Vec<OrtOwnedTensor<f32, IxDyn>> = session.run(input_names.iter().zip(input_tensor_dyn.iter()))?;
    
//     // let inputs: Vec<(&str, &ndarray::ArrayD<f32>)> = vec![
//     //     (&input_names[0], &input_tensor_dyn[0])
//     // ];
//     // let outputs: Vec<OrtOwnedTensor<f32, IxDyn>> = session.run(inputs)?;

//     let input_values: Vec<Value> = input_tensor_dyn
//         .iter()
//         .map(|arr| NdArrayTensor::from_array(arr.clone()).unwrap().into())
//         .collect();

//     let outputs: Vec<OrtOwnedTensor<f32, IxDyn>> = session.run_with_ort_inputs(&input_names, &input_values)?;


//     // Step 5: Use results
//     let output = &outputs[0];
//     println!("Output shape: {:?}", output.shape());
//     println!("First 10 output values: {:?}", &output.view().iter().take(10).collect::<Vec<_>>());

//     Ok(())
// }

// fn run_inference() -> Result<(), Box<dyn std::error::Error>> {
// //fn run_inference(mel: &Array3<f32>) -> Result<(), Box<dyn std::error::Error>> {
//     // Load ONNX model
//     let environment = Arc::new(Environment::builder().with_name("inference").build()?);
//     let mut session = environment
//         .new_session_builder()?
//         .with_model_from_file("efficientnet_b1_export.onnx")?;

//     // Load the mel spectrogram (must be shape: [1, 256, 431])
//     let mel: ArrayD<f32> = read_npy("../mel_outputs/test/HSN_001_20150708_061805_000_005_mel.npy")?;
    
//     // Reshape to [1, 1, 256, 431]
//     //println!("Raw shape of mel: {:?}", mel.shape());
//     let mel = mel.into_dimensionality::<ndarray::Ix3>()?; // from [1, 256, 431]
//     let mel = mel.insert_axis(ndarray::Axis(0)); // now [1, 1, 256, 431]
//     // Convert to dynamic dimensionality (IxDyn)
//     let input_tensor: Vec<ArrayD<f32>> = vec![mel.into_dyn()];
//     //println!("Final input shape: {:?}", input_tensor[0].shape());
//     // Wrap in a vector since ONNX expects Vec<ArrayD<T>> as input
//     //let input_tensor = vec![mel];

//     // Run inference
//     let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor)?;

//     // Inspect result
//     let output_tensor = &outputs[0];

//     let label_names = vec![
//         "gcrfin", "whcspa", "amepip", "sposan", "rocwre", "brebla", "daejun", "foxspa",
//         "clanut", "moublu", "casfin", "mallar3", "herthr", "amerob", "yerwar", "yelwar",
//         "dusfly", "mouchi", "orcwar", "warvir", "norfli"
//     ];
//     // Convert [1, 21] -> [21]
//     let logits = output_tensor
//         .view()
//         .to_owned()
//         .into_dimensionality::<ndarray::Ix2>()?;
//     let logits_row = logits.row(0);

//     // Print each class index and its logit score
//     // for (i, score) in logits_row.iter().enumerate() {
//     //     println!("Class {}: {:.4}", i, score);
//     // }
//     for (i, score) in logits_row.iter().enumerate() {
//         println!("Class {} ({}) : {:.4}", i, label_names[i], score);
//     }


//     Ok(())
// }


fn run_inference() -> Result<(), Box<dyn std::error::Error>> {
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

    // Directory containing mel files
    let mel_dir = Path::new("../mel_outputs/test");

    for entry in fs::read_dir(mel_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|ext| ext == "npy").unwrap_or(false)
            && path.file_name().unwrap_or_default().to_str().unwrap_or("").contains("_mel")
        {
            let mel: ArrayD<f32> = read_npy(&path)?;

            // println!(
            //     "min: {:.4}, max: {:.4}, mean: {:.4}",
            //     mel.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            //     mel.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            //     mel.iter().sum::<f32>() / mel.len() as f32
            // );

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

// fn log_system_usage(phase: &str) {
//     let mut sys = System::new_all();
//     sys.refresh_all();

//     println!("\n=== System Usage during: {} ===", phase);
//     println!("Total Memory: {} MB", sys.total_memory() / 1024);
//     println!("Used Memory:  {} MB", sys.used_memory() / 1024);
//     println!("CPU Usage:    {:.2}%", sys.global_cpu_info().cpu_usage());
//     println!("======================================\n");
// }

fn log_system_usage(phase: &str) {
    let mut sys = System::new_all();
    sys.refresh_all();

    let now = Local::now();
    // let log = format!(
    //     "\n=== System Usage during: {} ===\n\
    //      Total Memory: {} MB\n\
    //      Used Memory:  {} MB\n\
    //      CPU Usage:    {:.2}%\n\
    //      ======================================\n",
    //     phase,
    //     sys.total_memory() / 1024,
    //     sys.used_memory() / 1024,
    //     sys.global_cpu_info().cpu_usage()
    // );
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