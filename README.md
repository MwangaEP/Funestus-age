Up to date files for testing:

**Data folder contains**
1. Train data (_An. funestus_)
2. Test data (_An. funestus_)

**Code folder contains**
1. standard_ml_fun_age_publication_fullwn.py : XGBoost classifier trained with all spectra features/wavenumbers
2. standard_ml_fun_age_publication_fselected_wn.py : XGBoost classifier retrained with 100 top features/wavenumbers influencing the prediction of the initial model
3. MLP_fun_age_XGB_feat_importances.py : Multilayer perceptron trained with fewer features (n = 100, selected using XGBoost feature importances)
4. MLP_PCA.py : Multilayer perceptron trained with 8 principal components, same as Mwanga et al., 2023; https://doi.org/10.1186/s12859-022-05128-5
