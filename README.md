# acoustic-species-inference-conversion
## by Siya Kamboj

### Abstract
To monitor biodiversity in a non-invasive way, researchers deploy acoustic loggers in natural habitats, generating terabytes of audio data. Manually labeling this much data is a major bottleneck for environmentalists’ ability to track biodiversity. A cross-platform desktop application, with efficient, on-device inference for species identification, can streamline this process. This paper focuses on converting Python-based inference pipelines into Rust, a programming language known for its performance and safety. By leveraging Burn and onnxruntime, we integrate machine learning inference directly into a Rust-based Electron app. This approach eliminates the need for a Python runtime or local web server, significantly improving performance, portability, and ease of deployment. We benchmarked the Rust-based inference against its Python counterpart and observed reduced time and comparable precision. These results demonstrate Rust’s viability for building high-performance, field-ready tools for scalable biodiversity monitoring.


### Important Links
- Midpoint Presentation: https://docs.google.com/presentation/d/14yESbDrgv_Fm4Z0y2W2DdVJx4XK_dFmQiILIhqsL5zo/edit?usp=sharing
- Final Presentation: https://docs.google.com/presentation/d/1KW4EVF14d5Ywq7Dlyfhr5BObpYU8zk9Tii_ZmJp0y0w/edit?usp=sharing

### Run the Code
- cd pyha-analyzer-2.0
- source .venv/bin/activate
- export PYTHONPATH="/Users/siyakamboj/Downloads/acoustic-species-inference-conversion/pyha-analyzer-2.0/burn_app/pyha_analyzer:$PYTHONPATH"
- Run the following commands in the terminal. Do this for every new terminal session. 
    - export ORT_STRATEGY=system
    - export ORT_LIB_LOCATION=/opt/homebrew/lib
- cd burn_app
- cargo run


### Important Notes
It should be noted that since pyha_analayzer_2.0 is a clone of a private github repo, github prevents you from being able to see the file contents. Therefore, I placed a copy of my rust code, main.rs, in the home directory. Even though this code cannot be run in a meaningful way from the home directory, it does document the work that was done. 

To understand the full file structure, please consult this image: 
![image](images/FileStructure.png)





