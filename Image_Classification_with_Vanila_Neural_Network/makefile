all:
	@clear 	
	python final.py --lr 0.0001 --momentum 0.5 --num_hidden 3 --sizes 300,300,300 --activation sigmoid --loss ce --opt adam --batch_size 20 --anneal true --save_dir sav/ --expt_dir expt/ --train data/train.csv --test data/test.csv --val data/val.csv
	python final.py --lr 0.0001 --momentum 0.5 --num_hidden 3 --sizes 300,300,300 --activation sigmoid --loss ce --opt nag --batch_size 20 --anneal true --save_dir sav/ --expt_dir expt/ --train data/train.csv --test data/test.csv --val data/val.csv