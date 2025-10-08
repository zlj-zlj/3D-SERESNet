# 3D-SERESNet

3D-SERESNet is a epileptic seizure prediction model. Implementation of paper - [Robust Epileptic Seizure Prediction: A 3D-SERESNet Framework for Patient-Specific and Multi-Patient Generalization]



## Requirements



``` shell
numpy 
pyedflib
scipy
torch
```

</details>

## datas
CHB-MIT Dataset: Official download link: https://physionet.org/content/chbmit/1.0.0/  


## run

``` shell

python predict.py --model model.pt --i inputdir --0 outputdir 

```
