>These are the codes for the model TranSIV from the paper entitled "Learning and Transferring Social and Item Visibilities for Personalized Recommendation" in CIKM 2017. They are modified from the codes of ExopoMF.

>If you have any questions, please contact z-m AT tsinghua.edu.cn or jackielinxiao AT gmail.com

=TranSIV

-Models Implemented:
TranSIV: Transferring Social and Item Visibilities

-Files
*pro_epinion.py
*new_semf_bx.py
*rec_new.py
*run_new.py
*README.md

-Requirements
*TensorFlow
*Numpy
*Scipy
*Pandas
*joblib

-Format of the Input Data
*'ratings.txt': Input ratings in the form of "user id, item id, 1";
*'test_ratings.txt': Input testing ratings in the form of "user id, item id, 1";
*'train_ratings.txt': Input training ratings in the form of "user id, item id, 1";
*'trust.txt': Input social connections in the form of "user id, user id , 1"
-Usage
1. run pro_epinion.py to generate the input data;
2. then run python run_new.py to fit model TranSIV and make predictions