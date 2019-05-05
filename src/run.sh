python build_ext_data.py
python train1.py -train valid=0
python train2.py -train valid=0
python train1.py -predict
cp bestmodel.h5 bestmodel0.h5
cp submission.csv submission0.csv
rm bestmodel.h5

python build_ext_data.py
python train1.py -train valid=1
python train2.py -train valid=1
python train1.py -predict
cp bestmodel.h5 bestmodel1.h5
cp submission.csv submission1.csv
rm bestmodel.h5

python build_ext_data.py
python train1.py -train valid=2
python train2.py -train valid=2
python train1.py -predict
cp bestmodel.h5 bestmodel2.h5
cp submission.csv submission2.csv
rm bestmodel.h5

python build_ext_data.py
python train1.py -train valid=3
python train2.py -train valid=3
python train1.py -predict
cp bestmodel.h5 bestmodel3.h5
cp submission.csv submission3.csv
rm bestmodel.h5

python build_ext_data.py
python train1.py -train valid=4
python train2.py -train valid=4
python train1.py -predict
cp bestmodel.h5 bestmodel4.h5
cp submission.csv submission4.csv
rm bestmodel.h5

python build_ext_data.py
python train1.py -train valid=5
python train2.py -train valid=5
python train1.py -predict
cp bestmodel.h5 bestmodel5.h5
cp submission.csv submission5.csv
rm bestmodel.h5

python build_ext_data.py
python train1.py -train valid=6
python train2.py -train valid=6
python train1.py -predict
cp bestmodel.h5 bestmodel6.h5
cp submission.csv submission6.csv
rm bestmodel.h5

python build_ext_data.py
python train1.py -train valid=7
python train2.py -train valid=7
python train1.py -predict
cp bestmodel.h5 bestmodel7.h5
cp submission.csv submission7.csv
rm bestmodel.h5

python build_ext_data.py
python train1.py -train valid=8
python train2.py -train valid=8
python train1.py -predict
cp bestmodel.h5 bestmodel8.h5
cp submission.csv submission8.csv
rm bestmodel.h5

python build_ext_data.py
python train1.py -train valid=9
python train2.py -train valid=9
python train1.py -predict
cp bestmodel.h5 bestmodel9.h5
cp submission.csv submission9.csv
rm bestmodel.h5

rm submission.csv
python avg.py
