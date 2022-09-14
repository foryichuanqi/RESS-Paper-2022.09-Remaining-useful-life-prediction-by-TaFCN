# Remaining-useful-life-prediction-by-LM-and-TaFCN-in-CMAPSS-dataset
Trend attention fully convolutional network for remaining useful life estimation in the turbofan engine of CMAPSS dataset. Signal selection, Attention mechanism, and Interpretability of deep learning are explored.

# Easy to run successfully
To make code easy to run successfully, we debug the files carefully. Generally speaking, if environments are satisfied, you can directly run all the xxx.py files inside after decompressing the compressed package without changing any code.

(1) Unzip Remaining_useful_life_prediction_by_LM_and_TaFCN_in_CMAPSS_dataset-main.zip

(2) Rename the two folders by deleting "-main". "-" may cause path errors. Note that there are two folders that need to be renamed, they are the parent folder
and its subfolder.

"Remaining_useful_life_prediction_by_LM_and_TaFCN_in_CMAPSS_dataset-main" to "Remaining_useful_life_prediction_by_LM_and_TaFCN_in_CMAPSS_dataset"

(3) Run any xxx.py directly

# Paper of Code and Citation
(1) To better understand our code, please read our paper.

    Paper: Trend attention fully convolutional network for remaining useful life estimation
    The website of the paper：https://www.sciencedirect.com/science/article/pii/S0951832022002356 

(2) Please cite this paper and the original source of the dataset when using the code for academic purposes.

    GB/T 7714: 

    Fan L, Chai Y, Chen X. Trend attention fully convolutional network for remaining useful life estimation[J]. Reliability Engineering & System Safety, 2022: 108590.

BibTex:

    @article{fan2022trend,
  title={Trend attention fully convolutional network for remaining useful life estimation},
  author={Fan, Linchuan and Chai, Yi and Chen, Xiaolong},
  journal={Reliability Engineering \& System Safety},
  pages={108590},
  year={2022},
  publisher={Elsevier}
}


# Relationship between Code and Paper

 (1) Section 2.2. Loss boundary to mapping ability
 :code\signal selection   

 (2) Section 2.3. Trend attention fully convolutional network
 :code\main(grid_FD_multi_channel_one_FCN_RUL_TaNet_attention_1out_all_train_for_test).py

 (3) Section  4. Interpretability
 :code\figure\interpretability_analysis.py

 (4) Fig. 6. Attention analysis of TaNet.
 :code\figure\interpretability_TaNet_analysis.py

 (5) Fig. 7. Accumulated prediction error over RUL.
 :code\figure\prediction_error.py

 (6) Fig. 5. Wilcoxon signed rank test comparison of eight combinations
 :code\figure\heatmap_p.py   and   code\table\wilcxon.py


# Environment and Acknowledgement:

(1) Environment:

    tensorflow-gpu            1.15.0
    
    keras                     2.2.4
    
    scipy                     1.5.2
    
    pandas                    1.0.5
    
    numpy                     1.19.1


(2) Acknowledgement: 
   Thanks for the following references sincerely.
   
   github：https://github.com/Vardoom/PredictiveMaintenanceNASA/blob/master/preprocess.ipynb
   
   github：https://github.com/schwxd/LSTM-Keras-CMAPSS
   
   github：https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
