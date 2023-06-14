# RPBP
 
This directory contains the source code of unpublished articles.  
It is for review only. Do not propagate.Thank you very much.

## Environment Preparation
    conda create -n rpbp python=3.7
    conda activate rpbp
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
    pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu102.html
    pip install torch-sparse==0.6.15 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu102.html
    pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu102.html
    pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu102.html
    pip install torch-geometric==2.1.0.post1
    conda install rdkit -c rdkit
    conda install joblib
    conda install pyyaml
    conda install textdistance
    pip install OpenNMT-py==2.3.0
    pip install imbalanced-learn
    pip install ipdb
    conda install networkx

## Workflow
### Step 1:  Stage-One Data preparation  
  

    cd preprocess
  * 1.1:Extraction the by-product from the reaction  
  

    python generate_byproduct.py  

  * 1.2: Data processing  
  

    python generate_product_pkl.py
    python generate_product_pkl.py --with_class  


### Step 2: Train model of stage-one and infer by-product  

    cd ..
  * 2.1: Predict by-product  
 
        python predict_byproduct.py --epoch_sample True
        python predict_byproduct.py --epoch_sample True --with_class 
    
### Step 3: Stage-Two Data preparation
    cd preprocess
  * 3.1: Generate data for the training stage two model
  
        python generate_inputs_for_stage_two.py 
  * 3.2: Process the prediction results of the stage-one to input the trained stage-two model  
  
        python generate_inputs_for_stage_two.py --mode test
        python generate_inputs_for_stage_two.py --mode test --with_class


### Step 4: Train model of stage-two and translate reactants  
    cd ..
    mkdir -p USPTO-50K/results/stage_two/without_class
    mkdir -p USPTO-50K/results/stage_two/with_class
  * 4.1: Build vocab  
  
        onmt_build_vocab --config  USPTO-50K/yaml/without_class/PtoR-50K-aug20-build_vocab.yml
        onmt_build_vocab --config  USPTO-50K/yaml/with_class/PtoR-50K-aug20-build_vocab.yml
  * 4.2: Train model of stage-two  
  
        onmt_train --config USPTO-50Kyaml/without_class/PtoR-50K-aug20-config.yml
        onmt_train --config USPTO-50Kyaml/with_class/PtoR-50K-aug20-config.yml
  * 4.3: Average model  
  
        sh USPTO-50K/yaml/without_class/PtoR-50K-aug20-average.sh
        sh USPTO-50K/yaml/with_class/PtoR-50K-aug20-average.sh
  * 4.4: Translate reactants  
  
        onmt_translate --config USPTO-50K/yaml/without_class/PtoR-50K-aug20-translate.yml -with_score
        onmt_translate --config USPTO-50K/yaml/with_class/PtoR-50K-aug20-translate.yml -with_score	  


### Step 5: Score
    python score.py --stage_one_scores USPTO-50K/results/stage_one/without_class/test_top10_scores.txt --predictions USPTO-50K/results/stage_two/without_class/average_model_56-60-results.txt --targets USPTO-50K/dataset/stage_two/without_class/top10_tgt-test.txt
    python score.py --stage_one_scores USPTO-50K/results/stage_one/with_class/test_top10_scores.txt --predictions USPTO-50K/results/stage_two/with_class/average_model_56-60-results.txt --targets USPTO-50K/dataset/stage_two/with_class/top10_tgt-test.txt
     


## Acknowledgement 

Graphretro: https://github.com/vsomnath/graphretro  
GraphSmote: https://github.com/TianxiangZhao/GraphSmote  
OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py  
Root-aligned SMILES: https://github.com/otori-bird/retrosynthesis
