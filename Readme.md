# RPSP
 
This directory contains the source code of unpublished articles.  
It is for review only. Do not propagate.Thank you very much.

* Step 1:  Stage One Data preparation  
cd preprocess
	<!-- Extraction of side products from the reaction -->
  * 1.1: python generate_side_product.py 
  <!-- Data processing -->
  * 1.2: python generate_product_pkl.py
    * python generate_product_pkl.py --with_class  


* Step 2:  Train model of stage one and infer Side Product  
cd ..
    <!-- Predict side product -->
  * 2.1: python predict_side_product.py --epoch_sample True
    * python predict_side_product.py --epoch_sample True --with_class 
    

* Step 3: Stage Two Data preparation
cd preprocess
    <!-- Generate data for the training stage two model -->
  * 3.1: python generate_inputs_for_stage_two.py 
    <!-- Process the prediction results of the stage one to input the trained stage two model -->
  * 3.2: python generate_inputs_for_stage_two.py --mode test
    * python generate_inputs_for_stage_two.py --mode test --with_class


* Step 4: Train model of stage two and translate reactants  
cd ..
    <!-- Build vocab -->
  * 4.1: onmt_build_vocab --config  yaml/without_class/PtoR-50K-aug20-build_vocab.yml
    * onmt_build_vocab --config  yaml/with_class/PtoR-50K-aug20-build_vocab.yml
    <!-- Train model of stage two -->
  * 4.2: onmt_train --config yaml/without_class/PtoR-50K-aug20-config.yml
    * onmt_train --config yaml/with_class/PtoR-50K-aug20-config.yml
    <!-- Average model -->
  * 4.3: sh yaml/without_class/PtoR-50K-aug20-average.sh
    * sh yaml/with_class/PtoR-50K-aug20-average.sh
    <!-- Translate reactants -->
  * 4.4: onmt_translate --config yaml/without_class/PtoR-50K-aug20-translate.yml -with_score
    * onmt_translate --config yaml/with_class/PtoR-50K-aug20-translate.yml -with_score	  


* Step 5: Score
  * 5.1: python score.py --stage_one_scores results/stage_one/without_class/test_top10_scores.txt --predictions results/stage_two/without_class/average_model_56-60-results.txt --targets results/stage_one/without_class/top10tgt_test.txt
    * python score.py --stage_one_scores results/stage_one/with_class/test_top10_scores.txt --predictions results/stage_two/with_class/average_model_56-60-results.txt --targets results/stage_one/with_class/top10tgt_test.txt
     


## Acknowledgement 

Graphretro: https://github.com/vsomnath/graphretro  
GraphSmote: https://github.com/TianxiangZhao/GraphSmote  
OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py  
Root-aligned Smiles: https://github.com/otori-bird/retrosynthesis