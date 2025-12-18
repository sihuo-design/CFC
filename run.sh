# llm_name: e5 lr 0.01
# llm_name: ST lr 0.1
# llm_name: llama lr 0.001 


for method in llm
do
    for dataset in cora
    do
        for lr in 0.01
        do
            for fine_method in denoising_mixup
            do 
                for seed in 100 101 102 103 104
                do  
                    python main.py --dataset $dataset\
                                    --model_type $method\
                                    --lr $lr\
                                    --seen_unseen_classifier llm_test\
                                    --fine_method $fine_method\
                                    --seed $seed
                done
            done
        done
    done
done

